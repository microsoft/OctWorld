"""Offline validation test bench for OctWorld Wan FramePack.

Loads one or more checkpoints saved by the training :class:`Checkpointer`,
rebuilds the :class:`WanMultiAdapterPipeline` on a single GPU (no FSDP), and for
a fixed set of data samples runs flow-match sampling, VAE-decodes both the
sampled (pred) and ground-truth (gt) next-chunk latents, and reports the
LPIPS / PSNR / SSIM between the generated frames (the last ``gen_frames*4``
pixel frames).

Every sample is evaluated with the *same* seed and sampling flow across all
checkpoint steps so results are directly comparable.

The training pipeline is NOT touched — this is a separate inference entry point.

Example::

    python -m octworld.training.wan.validate \\
        --run_dir <run-dir> \
        --step 3000,4000,5000 \\
        --data_path <dataset-root> \
        --num_samples 50

Output layout (``exp_name`` is the basename of ``run_dir``)::

    debugs/<exp_name>_val/step_<N>/sample_<i>/pred.mp4
    debugs/<exp_name>_val/step_<N>/sample_<i>/gt.mp4
    debugs/<exp_name>_val/step_<N>/sample_<i>/compare.mp4   (gt | pred side-by-side)
    debugs/<exp_name>_val/step_<N>/sample_<i>/metric.json
    debugs/<exp_name>_val/step_<N>/summary.json            (means over samples)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from octworld.datasets import WanFramePackDataset
from octworld.models.wan.modules.vae import WanVAE
from octworld.pipelines.wan.wan_pipeline import WanMultiAdapterPipeline
from octworld.training.checkpoint import (
    MODEL_CHECKPOINT,
    get_latest_checkpoint_folder,
)
from octworld.training.wan.octworld_wan_trainer import _build_nested_config
from utils.loss import lpips as lpips_metric
from utils.loss import psnr as psnr_metric
from utils.loss import ssim as ssim_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wan.validate")

# History/gen layout produced by WanFramePackDataset:
#   latents = [anchor(1) | history(19) | gen(gen_frames)]
N_HISTORY_TOTAL = 20


# ----------------------------------------------------------------------
# Pipe construction + checkpoint load
# ----------------------------------------------------------------------
def _is_dcp_dir(path: str) -> bool:
    """True if ``path`` is a sharded DCP checkpoint dir (has a ``.metadata``)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, ".metadata"))


def _load_model_state_dict(ckpt_ref: str) -> dict:
    """Load a *full* model state_dict from either checkpoint layout.

    Two on-disk layouts are handled transparently:

    * **Single full .pt** (e.g. ``init/model.pt``): a plain ``torch.save`` of
      the model state_dict — loaded directly.
    * **Sharded DCP dir** (resume checkpoints written by ``Checkpointer.save``
      via ``dcp.save``): a directory of ``*.distcp`` shards + ``.metadata``. It
      is read with an empty-state-dict planner and ``no_dist=True``, which
      reassembles every shard into a single full-tensor state dict on this one
      process — no torchrun / process group required, so a checkpoint saved on
      N GPUs loads fine on 1. The training state dict is nested as
      ``{"model": ..., "optim": ...}``; only the ``"model"`` sub-dict is
      returned (the optimizer is not needed for inference).
    """
    if _is_dcp_dir(ckpt_ref):
        log.info(f"[ckpt] loading sharded DCP checkpoint from {ckpt_ref}")
        full: dict = {}
        _load_state_dict(
            full,
            storage_reader=dcp.FileSystemReader(ckpt_ref),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        if "model" not in full:
            raise KeyError(
                f"DCP checkpoint {ckpt_ref} has no 'model' key "
                f"(top-level keys: {list(full)[:5]})")
        return full["model"]

    log.info(f"[ckpt] loading full-state .pt from {ckpt_ref}")
    return torch.load(ckpt_ref, map_location="cpu", weights_only=True)


def build_pipe_from_ckpt(
    pipe_config: dict, ckpt_path: str, device: torch.device, dtype: torch.dtype,
) -> WanMultiAdapterPipeline:
    """Build the pipe and load a training checkpoint (single GPU, no FSDP).

    Only the weight-allocating ``build_training_model`` runs under the meta
    device — the pipe constructor itself stays on a real device so its
    ``scheduler`` (and other host-side state) is not meta-allocated. The meta
    params are then replaced in-place by the checkpoint's real tensors via
    ``load_state_dict(assign=True)``, avoiding a redundant full copy of the
    14B weights.
    """
    pipe = WanMultiAdapterPipeline()  # real device: scheduler stays usable
    with torch.device("meta"):        # only the big weight allocation
        pipe.build_training_model(
            config=_build_nested_config(pipe_config),
            use_gradient_checkpointing=False,
        )

    sd = _load_model_state_dict(ckpt_path)
    # assign=True: meta params have no storage, so replace them with the
    # checkpoint's real tensors instead of copy_-ing into them.
    missing, unexpected = pipe.load_state_dict(sd, assign=True, strict=False)
    if missing:
        log.info(f"[ckpt] {len(missing)} missing keys (e.g. {missing[:5]})")
    if unexpected:
        log.info(f"[ckpt] {len(unexpected)} unexpected keys (e.g. {unexpected[:5]})")

    pipe.materialize()  # rebuild non-persistent RoPE freqs (not in state_dict)
    pipe = pipe.to(device=device, dtype=dtype).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in pipe.parameters()) / 1e9
    log.info(f"[ckpt] pipe ready: {n_params:.2f}B params on {device} ({dtype})")
    return pipe


# ----------------------------------------------------------------------
# Video helpers (decode → uint8 frames → mp4)
# ----------------------------------------------------------------------
def _latents_to_video(vae: WanVAE, latent: torch.Tensor, device) -> np.ndarray:
    """Decode a single (16, T, H, W) latent to (F, H*8, W*8, 3) uint8 frames."""
    z = latent.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        video = vae.decode([z])[0].float().cpu()  # (3, T*4, H*8, W*8) in [-1, 1]
    video = (video.clamp(-1, 1) + 1.0) / 2.0       # → [0, 1]
    video = video.permute(1, 2, 3, 0).numpy()      # (F, H, W, 3)
    return (video * 255.0).clip(0, 255).astype(np.uint8)


def _save_mp4(frames_thwc_uint8: np.ndarray, path: str, fps: int = 16):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, list(frames_thwc_uint8), fps=fps, quality=8)
    log.info(f"  saved {path}  shape={frames_thwc_uint8.shape}")


def _save_compare_mp4(gt: np.ndarray, pred: np.ndarray, path: str, fps: int = 16):
    """Save a side-by-side (gt | pred) comparison video.

    ``gt`` and ``pred`` are (F, H, W, 3) uint8 arrays of equal length/shape.
    """
    f = min(gt.shape[0], pred.shape[0])
    combined = np.concatenate([gt[:f], pred[:f]], axis=2)  # concat along width
    _save_mp4(combined, path, fps=fps)


# ----------------------------------------------------------------------
# Metrics (LPIPS / PSNR / SSIM) over the generated frames
# ----------------------------------------------------------------------
def _compute_metrics(gt: np.ndarray, pred: np.ndarray, device) -> dict:
    """LPIPS / PSNR / SSIM between two (F, H, W, 3) uint8 videos.

    Uses the shared ``utils.loss`` implementations (tensors in [0, 1],
    shape (F, 3, H, W)). Returns a dict with the scalar metric values.
    """
    f = min(gt.shape[0], pred.shape[0])

    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x[:f]).to(device=device, dtype=torch.float32)
        return t.permute(0, 3, 1, 2) / 255.0  # (F, 3, H, W) in [0, 1]

    gt_t, pred_t = _to_tensor(gt), _to_tensor(pred)
    with torch.no_grad():
        psnr_val = float(psnr_metric(pred_t, gt_t).item())
        ssim_val = float(ssim_metric(pred_t, gt_t).item())
        lpips_val = float(lpips_metric(pred_t, gt_t, value_range=(0, 1)).item())

    return {
        "lpips": lpips_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "num_frames": int(f),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=False, default=None, metavar="<run-dir>",
                   help="Run dir containing the ckpts/ folder "
                        "(i.e. an <exp_dir>/<tag> path).")
    p.add_argument("--ckpt_path", default=None,
                   help="Directly evaluate a single model state_dict .pt file "
                        "(e.g. an <exp>/init/model.pt), bypassing the "
                        "run_dir/ckpts/<step>/ layout. When given, --run_dir "
                        "and --step are optional.")
    p.add_argument("--ckpt_label", default=None,
                   help="Label for the --ckpt_path run, used for the output "
                        "subdir (step_<label>) and logging. Defaults to the "
                        "ckpt filename stem (e.g. 'model' for init/model.pt).")
    p.add_argument("--compare", type=str, default=None,
                   help="Compare-only mode: skip sampling and just scan this "
                        "validation dir for step_*/sample_*/metric.json, "
                        "average each metric per step and print a table. "
                        "Incomplete steps show how many samples are done.")
    p.add_argument("--pipe_config",
                   default="debugs/dl3dv-octworld-wan-wan14b_i2v-b64/pipe_config.json",
                   help="Path to the pipe_config.json manifest.")
    p.add_argument("--neg_text_embed", metavar="<embedding-path>",
                   default=None,
                   help="Path to neg_text_embed.pt (only used when cfg_scale != 1.0).")
    p.add_argument("--step", type=str, default=None,
                   help="Comma-separated checkpoint step(s) to evaluate, e.g. "
                        "'3000,4000,5000'. Default: latest under ckpts/.")
    p.add_argument("--vae_path",
                   default="model_weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
                   help="Path to the Wan VAE weights for decoding.")
    p.add_argument("--data_path", metavar="<dataset-root>",
                   default=None,
                   help="TARGET_PATH root for WanFramePackDataset.")
    p.add_argument("--metadata_path", default="train_metadata.csv",
                   help="Metadata CSV (relative to data_path).")
    p.add_argument("--num_samples", type=int, default=50,
                   help="Number of data-batch samples to generate.")
    p.add_argument("--gen_frames", type=int, default=20,
                   help="Generation latent frames (chunk size).")
    p.add_argument("--num_steps", type=int, default=35,
                   help="Number of euler denoising steps.")
    p.add_argument("--shift", type=float, default=3.0,
                   help="Flow-match shift (480p uses 3.0).")
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 disables CFG "
                        "(single forward/step, no negative prompt needed).")
    p.add_argument("--decode_history_frames", type=int, default=4,
                   help="Real history latent frames prepended before decoding "
                        "for temporal continuity (non-first chunks). First "
                        "chunks always prepend a single frame. The decoded "
                        "history pixels are KEPT at the start of each video.")
    p.add_argument("--noise_aug_level", type=int, default=None,
                   help="Fixed history noise-augmentation bucket level for "
                        "inference (0..10). Omit to disable augmentation.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed: shuffles the dataset order and seeds "
                        "each per-case sampling (offset by the case index).")
    p.add_argument("--out_dir", default=None,
                   help="Output dir. Default debugs/<exp_name>_val/ "
                        "(exp_name = basename of run_dir).")
    return p.parse_args()


def _resolve_steps(arg_step: str | None, ckpt_root: str) -> list[int]:
    """Parse the ``--step`` arg into a list of checkpoint steps to evaluate."""
    if arg_step is None:
        latest = get_latest_checkpoint_folder(ckpt_root)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found under {ckpt_root}")
        return [int(latest)]
    steps = [int(s) for s in arg_step.split(",") if s.strip() != ""]
    if not steps:
        raise ValueError(f"--step parsed to an empty list: {arg_step!r}")
    return steps


def _evaluate_sample(
    pipe, vae, batch, args, neg_text_embed, cfg_scale, device, dtype, case_seed,
):
    """Run sampling + decode for a single batch (B=1).

    Returns ``(pred_frames, gt_frames, info)`` where the frame arrays are
    (F, H, W, 3) uint8 and ``info`` holds meta + the eval window length.
    """
    # Fully control the RNG state per case (np / random / torch / cuda) so the
    # sampling flow is identical across checkpoint steps.
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    torch.cuda.manual_seed_all(case_seed)

    latents = batch["latents"].to(device=device, dtype=dtype)        # (1,16,T,H,W)
    rendered_latents = batch["rendered_latents"].to(device=device, dtype=dtype)  # (1,16,g,H,W)
    rendered_masks = batch["rendered_masks"].to(device=device, dtype=dtype)      # (1,4,g,H,W)
    raymap = batch["raymap"].to(device=device, dtype=dtype)          # (1,6,g,Hv,Wv)
    clip_features = batch["clip_features"].to(device=device, dtype=dtype)
    text_embed = [t.to(device=device, dtype=dtype) for t in batch["text_embed"]]

    history = latents[:, :, :N_HISTORY_TOTAL]   # (1,16,20,H,W) clean context
    B, C, _, H, W = latents.shape
    gen_frames = latents.shape[2] - N_HISTORY_TOTAL

    # History-condition noise augmentation. Like the world-gen inference
    # pipeline, we noise the history (except the anchor) + rendered latents
    # ONCE here and then hold that condition fixed across all denoising steps /
    # CFG branches (``prenoised_history=True``). This mirrors what the model saw
    # in training for a given level, WITHOUT re-sampling the augmentation noise
    # on every ``model_fn`` call (which would jitter the conditioning at
    # inference and corrupt guidance).
    noise_aug_level = getattr(args, "noise_aug_level", None)
    if noise_aug_level is not None:
        history_cond, rendered_cond, history_ts = (
            WanMultiAdapterPipeline.apply_history_noise_aug(
                history, rendered_latents, noise_aug_level,
                getattr(pipe.scheduler, "num_train_timesteps", 1000),
            )
        )
    else:
        history_cond, rendered_cond, history_ts = history, rendered_latents, None

    input_shared = dict(
        history_latents=history_cond,
        rendered_latents=rendered_cond,
        rendered_masks=rendered_masks,
        clip_features=clip_features,
        history_timestep=history_ts,
        prenoised_history=True,
    )
    input_pos = dict(context=text_embed, raymap=raymap)
    input_neg = dict(
        context=[neg_text_embed] * B if neg_text_embed is not None else text_embed,
        raymap=torch.zeros_like(raymap),
    )

    # Seeded initial noise for the gen segment.
    g = torch.Generator(device="cpu").manual_seed(case_seed)
    gen_noise = torch.randn(
        (B, C, gen_frames, H, W), generator=g, dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    meta = batch["meta"][0]
    is_first_chunk = bool(meta.get("is_first_chunk", False))
    # First chunk has no real history (only the anchor frame), so just prepend a
    # single frame; later chunks prepend a short real tail for continuity.
    k = 1 if is_first_chunk else args.decode_history_frames

    sampled = pipe.latent_sampling(
        gen_noise,
        cfg_scale=cfg_scale,
        cfg_merged=False,
        input_shared=input_shared,
        input_pos=input_pos,
        input_neg=input_neg,
    )  # (1, 16, gen_frames, H, W)

    # Prepend the last k real history latents for temporal continuity; the
    # decoded history pixels are kept at the start of the video.
    hist_tail = history[:, :, -k:] if k > 0 else history[:, :, :0]
    pred_frames = _latents_to_video(
        vae, torch.cat([hist_tail, sampled], dim=2)[0], device)
    gt_gen = latents[:, :, N_HISTORY_TOTAL:]  # (1, 16, gen_frames, H, W)
    gt_frames = _latents_to_video(
        vae, torch.cat([hist_tail, gt_gen], dim=2)[0], device)

    info = {
        "batch_name": meta.get("batch_name", "?"),
        "chunk_idx": int(meta.get("chunk_idx", -1)),
        "is_first_chunk": is_first_chunk,
        "gen_frames": int(gen_frames),
        "decode_history_frames": int(k),
        # Generated pixel frames = gen_frames * VAE temporal stride (4).
        "eval_frames": int(gen_frames * 4),
    }
    return pred_frames, gt_frames, info


def compare_metrics(val_dir: str):
    """Scan a validation dir and average each metric over every step.

    Looks for ``step_*/sample_*/metric.json`` under ``val_dir``, averages the
    LPIPS / PSNR / SSIM of each step and prints a comparison table. Steps that
    did not finish all their samples are flagged with their completed count.
    """
    import re

    step_names = sorted(
        (d for d in os.listdir(val_dir)
         if os.path.isdir(os.path.join(val_dir, d)) and d.startswith("step_")),
        key=lambda d: int(re.sub(r"\D", "", d) or 0),
    )
    if not step_names:
        raise FileNotFoundError(f"No step_*/ dirs found under {val_dir}")

    metric_keys = ("lpips", "psnr", "ssim")
    higher_better = {"lpips": False, "psnr": True, "ssim": True}
    rows = []
    for name in step_names:
        step_dir = os.path.join(val_dir, name)
        sample_dirs = [d for d in os.listdir(step_dir) if d.startswith("sample_")]
        records = []
        for d in sample_dirs:
            mpath = os.path.join(step_dir, d, "metric.json")
            if os.path.isfile(mpath):
                with open(mpath) as f:
                    records.append(json.load(f))
        row = {"step": name, "done": len(records), "total": len(sample_dirs)}
        for k in metric_keys:
            vals = [r[k] for r in records if k in r]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)

    header = (f"{'step':>12} | {'done':>9} | "
              + " | ".join(f"{k:>8}" for k in metric_keys))
    log.info(header)
    log.info("-" * len(header))
    for row in rows:
        flag = "" if row["done"] == row["total"] else "  (incomplete)"
        done_str = f"{row['done']}/{row['total']}"
        cells = [f"{row['step']:>12}", f"{done_str:>9}"]
        cells += [f"{row[k]:>8.4f}" for k in metric_keys]
        log.info(" | ".join(cells) + flag)

    log.info("-" * len(header))
    for k in metric_keys:
        valid = [r for r in rows if not np.isnan(r[k])]
        if valid:
            better = max if higher_better[k] else min
            best = better(valid, key=lambda r: r[k])
            arrow = "↑" if higher_better[k] else "↓"
            log.info(f"  best {k}{arrow}: {best['step']} ({best[k]:.4f})")


def main():
    args = parse_args()
    if args.compare:
        compare_metrics(args.compare)
        return

    if args.run_dir is None and args.ckpt_path is None:
        raise ValueError(
            "one of --run_dir or --ckpt_path is required unless --compare is given.")
    if args.data_path is None:
        raise ValueError("--data_path is required for validation sampling")
    if args.cfg_scale != 1.0 and args.neg_text_embed is None:
        raise ValueError(
            f"--cfg_scale={args.cfg_scale} requires --neg_text_embed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    if args.ckpt_path is not None:
        # Direct single-checkpoint mode: evaluate one model.pt as-is.
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"--ckpt_path not found: {args.ckpt_path}")
        label = args.ckpt_label or os.path.splitext(
            os.path.basename(args.ckpt_path))[0]
        steps = [label]
        ckpt_paths = {label: args.ckpt_path}
        # exp_name: prefer --run_dir basename; else the experiment dir inferred
        # from the ckpt path (…/<exp>/init/model.pt -> <exp>).
        if args.run_dir is not None:
            exp_name = os.path.basename(os.path.normpath(args.run_dir))
        else:
            exp_name = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt_path))))
    else:
        run_dir = args.run_dir
        ckpt_root = os.path.join(run_dir, "ckpts")
        steps = _resolve_steps(args.step, ckpt_root)

        # Validate every requested checkpoint exists up front. Each step dir is
        # either a sharded DCP dir (resume format) or a legacy single .pt.
        ckpt_paths = {}
        for s in steps:
            step_dir = os.path.join(ckpt_root, str(s))
            legacy_pt = os.path.join(step_dir, MODEL_CHECKPOINT)
            if _is_dcp_dir(step_dir):
                ckpt_paths[s] = step_dir
            elif os.path.isfile(legacy_pt):
                ckpt_paths[s] = legacy_pt
            else:
                raise FileNotFoundError(
                    f"No checkpoint for step {s}: neither a DCP dir "
                    f"({step_dir}/.metadata) nor {legacy_pt}")

        exp_name = os.path.basename(os.path.normpath(run_dir))

    base_out = args.out_dir or os.path.join(
        PROJECT_ROOT, "debugs", f"{exp_name}_val")
    os.makedirs(base_out, exist_ok=True)
    log.info(f"[setup] exp_name={exp_name} steps={steps}")
    log.info(f"[setup] base_out={base_out}")

    # pipe_config manifest (path is independent of run_dir).
    with open(args.pipe_config) as f:
        pipe_config = json.load(f)

    # Negative prompt (only when CFG is enabled); independent of the checkpoint.
    neg_text_embed = None
    cfg_scale = args.cfg_scale
    if cfg_scale != 1.0:
        neg_path = args.neg_text_embed
        if not os.path.isfile(neg_path):
            raise FileNotFoundError(
                f"--cfg_scale={cfg_scale} requires {neg_path} (neg_text_embed.pt)")
        neg_data = torch.load(neg_path, map_location="cpu", weights_only=False)
        neg_text_embed = neg_data["text_embed"].to(device=device, dtype=dtype)
        log.info(f"[cfg] cfg_scale={cfg_scale}, neg_embed={tuple(neg_text_embed.shape)}")
    else:
        log.info("[cfg] CFG disabled (cfg_scale=1.0)")

    # Dataset (same one the trainer uses). Uses the default rendered-warp
    # variants (online_kf5 + offline_kf5), matching training.
    dataset = WanFramePackDataset(
        target_path=args.data_path,
        metadata_path=args.metadata_path,
        gen_frames=args.gen_frames,
        load_raymap=True,
    )
    n = min(args.num_samples, len(dataset))
    log.info(f"[data] dataset size={len(dataset)}, evaluating {n} samples")

    # Fixed, seed-shuffled sample selection — identical for every step.
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:n]

    # VAE for decoding (shared across steps).
    log.info(f"[vae] loading VAE from {args.vae_path}")
    vae = WanVAE(vae_pth=args.vae_path, device=str(device), dtype=torch.float32)

    for s in steps:
        step_out = os.path.join(base_out, f"step_{s}")
        os.makedirs(step_out, exist_ok=True)
        log.info(f"[step {s}] building pipe from {ckpt_paths[s]}")

        pipe = build_pipe_from_ckpt(pipe_config, ckpt_paths[s], device, dtype)
        pipe.scheduler.set_timesteps(
            num_inference_steps=args.num_steps, shift=args.shift)
        log.info(f"[step {s}] {len(pipe.scheduler.timesteps)} sched steps, "
                 f"shift={args.shift}")

        per_sample = []
        for i, idx in enumerate(indices):
            # Same per-case seed across steps for an identical sampling flow.
            case_seed = args.seed + i
            batch = WanFramePackDataset.collate_fn([dataset[idx]])  # B=1

            pred_frames, gt_frames, info = _evaluate_sample(
                pipe, vae, batch, args, neg_text_embed, cfg_scale,
                device, dtype, case_seed)

            # Evaluate the metric over the generated tail only.
            ev = info["eval_frames"]
            metrics = _compute_metrics(
                gt_frames[-ev:], pred_frames[-ev:], device)

            log.info(
                f"[step {s}][{i+1}/{n}] {info['batch_name']} "
                f"chunk={info['chunk_idx']} first={info['is_first_chunk']} | "
                f"lpips={metrics['lpips']:.4f} psnr={metrics['psnr']:.2f} "
                f"ssim={metrics['ssim']:.4f}")

            sample_dir = os.path.join(step_out, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            _save_mp4(pred_frames, os.path.join(sample_dir, "pred.mp4"))
            _save_mp4(gt_frames, os.path.join(sample_dir, "gt.mp4"))
            _save_compare_mp4(gt_frames, pred_frames,
                              os.path.join(sample_dir, "compare.mp4"))

            record = {"sample_idx": i, "dataset_idx": int(idx),
                      "seed": case_seed, **info, **metrics}
            with open(os.path.join(sample_dir, "metric.json"), "w") as f:
                json.dump(record, f, indent=2)
            per_sample.append(record)

        # Step-level summary: mean of each metric over all samples.
        summary = {
            "step": s,
            "num_samples": len(per_sample),
            "lpips": float(np.mean([r["lpips"] for r in per_sample])),
            "psnr": float(np.mean([r["psnr"] for r in per_sample])),
            "ssim": float(np.mean([r["ssim"] for r in per_sample])),
            "samples": per_sample,
        }
        with open(os.path.join(step_out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"[step {s}] DONE | mean lpips={summary['lpips']:.4f} "
                 f"psnr={summary['psnr']:.2f} ssim={summary['ssim']:.4f} "
                 f"-> {step_out}")

        # Free the pipe before loading the next checkpoint.
        del pipe
        torch.cuda.empty_cache()

    log.info(f"[done] evaluated steps={steps} over {n} samples -> {base_out}")


if __name__ == "__main__":
    main()
