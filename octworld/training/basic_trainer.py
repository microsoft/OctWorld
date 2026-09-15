"""BasicTrainer — distribution-agnostic training loop scaffold.

Subclasses implement the abstract hooks (``build_pipe``, ``compute_loss``,
``prepare_dataloader``). Everything else — the main loop, logging, gradient
accumulation, grad-clip, TensorBoard writer — lives here and knows nothing
about FSDP or the concrete model.

The trainer owns ``self.pipe`` (an ``nn.Module`` populated by the subclass's
``build_pipe()``) but only ever touches it through ``self.compute_loss`` and
``self.pipe.parameters()`` / ``self.pipe.train()``.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from octworld.training.checkpoint import Checkpointer
from octworld.training.utils import (
    all_gather_data,
    build_mesh,
    dict_flatten,
    dict_reduce,
    dtype_from_str,
    metrics_to_python,
    move_to_device_and_cast,
)
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """``torch.optim`` constructor kwargs (``name`` selects the class)."""

    name: str = "AdamW"
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


class FSDPConfig(BaseModel):
    """Sharding / mixed-precision configuration for FSDP."""

    mesh_mode: str = "2d"  # "1d" full-shard / "2d" HSDP
    param_dtype: str = "bfloat16"


class LoggingConfig(BaseModel):
    """Logging / checkpoint cadence."""

    i_log: int = 10
    i_save: int = 500
    tb_subdir: str = "tb_logs"


class BaseTrainConfig(BaseModel):
    """Distribution-agnostic config block consumed by :class:`BasicTrainer`.

    Concrete trainers keep this as the ``basic`` section of their own top-level
    config (see ``octworld/training/wan/config/config.py``) and pass it to
    ``BasicTrainer.__init__``. ``exp_dir`` / ``init_ckpt_path`` are runtime
    paths injected by the entry script before construction.
    """

    exp_dir: str = ""
    init_ckpt_path: Optional[str] = None
    distribute_mode: str = "fsdp"

    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    grad_clip: float = 1.0
    seed: int = 0
    num_workers: int = 4

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class BasicTrainer(ABC):
    """Distribution-agnostic training loop scaffold."""

    def __init__(self, *, cfg: BaseTrainConfig):
        assert cfg.distribute_mode == "fsdp", "Only 'fsdp' distribute_mode is supported for now"
        self.cfg = cfg
        self.distribute_mode = cfg.distribute_mode
        self.exp_dir = cfg.exp_dir
        self.init_ckpt_path = cfg.init_ckpt_path
        self.step = 0
        self.pipe = None          # populated by subclass.build_pipe()
        self.optimizer = None     # populated by init_models_and_more()
        self.dataset = None       # populated by prepare_dataloader() via build_dataset()
        self.dataloader = None    # populated by prepare_dataloader()
        self.sampler = None       # populated by prepare_dataloader()
        self.writer = None        # populated by init_logger()
        self.mesh = None          # populated by init_models_and_more() (FSDP)
        self.checkpointer = Checkpointer(os.path.join(cfg.exp_dir, "ckpts"))
        self._train_log_fh = None  # populated by init_training_logger()

        # Auto-detect distributed environment. The entry script (train.py) calls
        # dist.init_process_group(backend="nccl") before constructing the trainer,
        # which reads RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT from the env
        # vars that torchrun injects (env:// rendezvous). So by the time we get
        # here the process group is already up when launched under torchrun.
        if dist.is_available() and dist.is_initialized():
            # Multi-GPU params. Prefer the torchrun-provided LOCAL_RANK /
            # LOCAL_WORLD_SIZE env vars: on multi-node, rank % device_count is
            # not guaranteed to equal the true local rank.
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_world_size = int(
                os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))
            self.local_rank = int(
                os.environ.get("LOCAL_RANK", self.rank % self.local_world_size))
            self.is_master = self.rank == 0
            self.distributed = True
        else:
            # Single-GPU / non-distributed params.
            self.world_size = 1
            self.local_world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True
            self.distributed = False

        # Bind this process to its GPU (must happen before any CUDA allocation).
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        # Per-rank seeding so data shuffling / augmentation differ across ranks.
        torch.manual_seed(cfg.seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed + self.rank)

        # Validate batch math: global = micro * grad_accum * world_size.
        expected = (cfg.micro_batch_size
                    * cfg.gradient_accumulation_steps * self.world_size)
        assert cfg.global_batch_size == expected, (
            f"batch math mismatch: global={cfg.global_batch_size} "
            f"!= micro({cfg.micro_batch_size}) * grad_accum"
            f"({cfg.gradient_accumulation_steps}) * world({self.world_size})"
        )

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses.
    # ------------------------------------------------------------------
    @abstractmethod
    def compute_loss(self, batch):
        """Return ``(loss: Tensor, metrics: dict)``."""
        ...

    @abstractmethod
    def build_pipe(self):
        """Populate ``self.pipe`` with an ``nn.Module``. Called inside a
        ``torch.device('meta')`` context by :meth:`init_models_and_more`."""
        ...

    @abstractmethod
    def build_dataset(self):
        """Return the training ``Dataset``. The base trainer wraps it in a
        ``DataLoader`` (collate / sampler / workers) inside
        :meth:`prepare_dataloader`. If the dataset exposes a ``collate_fn``
        attribute it is used as the loader's ``collate_fn``."""
        ...

    # ------------------------------------------------------------------
    # Dataloader construction (model-agnostic).
    # ------------------------------------------------------------------
    def prepare_dataloader(self):
        """Build ``self.dataset`` (via :meth:`build_dataset`) and wrap it in a
        ``DataLoader``. Uses a :class:`ResumableSampler` so the (epoch, idx)
        position can be checkpointed and restored for exact resume."""
        from torch.utils.data import DataLoader

        from octworld.training.resumable_sampler import ResumableSampler

        self.dataset = self.build_dataset()
        assert self.dataset is not None, "build_dataset() must return a Dataset"

        num_workers = self.cfg.num_workers
        # ResumableSampler handles both single-process and distributed cases
        # (it reads world_size/rank from the process group), so we always use it
        # and let the DataLoader defer shuffling to the sampler.
        self.sampler = ResumableSampler(
            self.dataset, shuffle=True, seed=self.cfg.seed, drop_last=True,
        )

        # Resume the sampler position from the latest checkpoint (if any). The
        # matching model/optimizer/step were already restored in
        # init_models_and_more(); this keeps the data stream aligned with them.
        if not self.checkpointer.is_empty():
            restored = self.checkpointer.load_sampler(self.sampler)
            if restored:
                self.print_and_log(
                    f"[data] resumed sampler at epoch={self.sampler.epoch} "
                    f"idx={self.sampler.idx}")
            else:
                self.print_and_log(
                    "[data] no sampler state in checkpoint; starting data "
                    "stream from epoch=0 idx=0")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.micro_batch_size,
            num_workers=num_workers,
            sampler=self.sampler,
            shuffle=False,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            drop_last=True,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Core model-loading recipe.
    # ------------------------------------------------------------------
    def init_models_and_more(self):
        # 1. meta-init pipe
        with torch.device("meta"):
            self.build_pipe()
        assert self.pipe is not None, "build_pipe() must populate self.pipe"

        # 2. apply FSDP (params remain on meta until the checkpoint load)
        self.training_dtype = dtype_from_str(self.cfg.fsdp.param_dtype)
        if self.distributed and self.distribute_mode == "fsdp":
            self.mesh = build_mesh(self.world_size, self.local_world_size, self.cfg.fsdp.mesh_mode)
            self.pipe.apply_fsdp(self.mesh, dtype=self.training_dtype)

        # 3. checkpoint load (DCP broadcast_from_rank0 materializes meta params)
        resume_step = None
        if not self.checkpointer.is_empty():
            self.checkpointer.load_model(self.pipe)
            resume_step = self.checkpointer.last_step
            self.print_and_log(f"[fsdp] resumed model from step {resume_step}")
        elif self.init_ckpt_path is not None:
            self.checkpointer.load_model_from_path(self.pipe, self.init_ckpt_path)
            self.print_and_log(f"[fsdp] loaded init checkpoint: {self.init_ckpt_path}")
        else:
            self.print_and_log("[fsdp] no checkpoint to load; params stay meta/empty")

        # 4. materialize non-persistent buffers (RoPE freqs)
        self.pipe.materialize()

        # 5. ensure everything lives on the target device
        self.pipe.to(device=self.device, dtype=self.training_dtype)

        # 5b. Apply deferred torch.compile now that params are loaded and
        # materialized on device. Compiling here (after the checkpoint load)
        # keeps the checkpoint's clean parameter FQNs intact — compiling earlier
        # inserts an ``_orig_mod.`` prefix that breaks the init-state-dict load.
        # torch.compile is lazy, so this is functionally identical to compiling
        # earlier. No-op for pipes without the hook.
        if hasattr(self.pipe, "maybe_compile_blocks"):
            self.pipe.maybe_compile_blocks()

        # 6. optimizer (params are already DTensors → FSDP-aware)
        opt_cfg = self.cfg.optimizer
        self.optimizer = getattr(torch.optim, opt_cfg.name)(
            self.pipe.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

        # 7. resume optimizer state
        if resume_step is not None:
            self.checkpointer.load_optim(self.pipe, self.optimizer)
            self.step = resume_step
            self.print_and_log(f"[fsdp] resumed optimizer from step {resume_step}")
            
    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Logging.
    # ------------------------------------------------------------------
    def init_training_logger(self):
        """Open a timestamped log file under ``<exp_dir>/training_logs/``.

        Only the master rank opens the file. All trainer output goes through
        :meth:`print_and_log`, which prints to stdout (so SLURM's own stdout
        redirection still works) and additionally appends to this file on rank0.
        """
        if not self.is_master:
            return
        log_dir = os.path.join(self.exp_dir, "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{ts}_log.txt")
        self._train_log_fh = open(log_path, "a")
        self.print_and_log(f"[log] training logs written to {log_path}")

    def print_and_log(self, *parts, flush: bool = True):
        """Print ``parts`` to stdout and append them to the training log file.

        Mirrors ``print`` (space-joined args). The file write only happens on
        the master rank, where ``self._train_log_fh`` is open; other ranks just
        print. Safe to call before :meth:`init_training_logger` (file write is
        skipped while the handle is ``None``).
        """
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=flush)
        if self._train_log_fh is not None:
            self._train_log_fh.write(msg + "\n")
            if flush:
                self._train_log_fh.flush()

    def init_logger(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        if self.is_master:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                os.path.join(self.exp_dir, self.cfg.logging.tb_subdir))

    def _log(self, loss: float, metrics: dict, t0: float, n_steps: int):
        now = time.time()
        speed = n_steps / max(1e-6, now - t0) * 3600  # steps/h
        time_elapsed = now - self._train_start_time
        remaining = self.cfg.max_steps - self.step
        eta = remaining / speed if speed > 0 else float('inf')
        max_mem = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if torch.cuda.is_available() else 0.0
        )
        columns = [
            f'Step: {self.step}/{self.cfg.max_steps} '
            f'({self.step / self.cfg.max_steps * 100:.2f}%)',
            f'Elapsed: {time_elapsed / 3600:.2f} h',
            f'Speed: {speed:.2f} steps/h',
            f'ETA: {eta:.2f} h',
            f'Max mem: {max_mem:.2f} MB',
            f'Loss: {loss:.4f}',
        ]
        description = '| '.join([c.ljust(25) for c in columns])
        self.print_and_log(description)
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, self.step)
            self.writer.add_scalar("train/ms_per_step", (now - t0) / max(1, n_steps) * 1000, self.step)
            # Flatten nested metric dicts (e.g. {'bin_0': {'mse': ...}}) so each
            # leaf becomes its own scalar (train/bin_0/mse).
            for k, v in dict_flatten(metrics or {}, sep="/").items():
                self.writer.add_scalar(f"train/{k}", v, self.step)
    # ------------------------------------------------------------------
    # Checkpoint I/O via Checkpointer.
    # ------------------------------------------------------------------
    def save_checkpoint(self, step: int):
        self.checkpointer.save(self.pipe, self.optimizer, step, sampler=self.sampler)

    # ------------------------------------------------------------------
    # Batch helpers.
    # ------------------------------------------------------------------
    def _next_batch(self, data_iter):
        # Advance the resumable sampler's per-rank position by one micro-batch
        # so checkpoints capture exactly how far the data stream has progressed.
        # All ranks step in lockstep (equal per-rank samples + drop_last), so
        # (epoch, idx) stays identical across ranks.
        try:
            batch = next(data_iter)
            if self.sampler is not None:
                self.sampler.idx += self.cfg.micro_batch_size
            return batch, data_iter
        except StopIteration:
            # Epoch boundary: advance to the next epoch (new shuffle) and reset
            # the intra-epoch offset before rebuilding the iterator.
            if self.sampler is not None:
                self.sampler.epoch += 1
                self.sampler.idx = 0
            data_iter = iter(self.dataloader)
            batch = next(data_iter)
            if self.sampler is not None:
                self.sampler.idx += self.cfg.micro_batch_size
            return batch, data_iter

    # ------------------------------------------------------------------
    # Memory profiling.
    # ------------------------------------------------------------------
    def _log_memory_footprint(self, tag: str):
        """Log CUDA memory currently held, before any training forward runs.

        Captures the static footprint of model params (+ optimizer state if
        already allocated) on this rank. Optimizer state for AdamW is created
        lazily on the first ``optimizer.step()``, so when called before the
        training loop this primarily reflects model parameters.
        """
        if not torch.cuda.is_available() or not self.is_master:
            return
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.print_and_log(
            f"[mem][{tag}] rank{self.rank} "
            f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"max_allocated={max_allocated:.2f}GiB"
        )

    # ------------------------------------------------------------------
    # Main training loop.
    # ------------------------------------------------------------------
    def run_training(self):
        assert self.pipe is not None, "init_models_and_more() must run first"
        assert self.dataloader is not None, "prepare_dataloader() must run first"
        self.pipe.train()

        # Static memory footprint (model + optimizer) before any forward pass.
        self._log_memory_footprint("pre-train")

        data_iter = iter(self.dataloader)
        ga = self.cfg.gradient_accumulation_steps
        max_steps = self.cfg.max_steps
        grad_clip = self.cfg.grad_clip
        i_log, i_save = self.cfg.logging.i_log, self.cfg.logging.i_save

        self._train_start_time = time.time()
        t0 = time.time()
        while self.step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss, micro_metrics = 0.0, []
            for _ in range(ga):
                batch, data_iter = self._next_batch(data_iter)
                batch = move_to_device_and_cast(batch, self.device, dtype=self.training_dtype)

                loss, metrics = self.compute_loss(batch)

                (loss / ga).backward()

                accum_loss += loss.detach().item()
                micro_metrics.append(metrics_to_python(metrics or {}))
            accum_loss /= ga
            # Reduce the per-microbatch nested metric dicts elementwise.
            accum_metrics = dict_reduce(
                micro_metrics, lambda x: float(np.mean(x))) if micro_metrics else {}

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.pipe.parameters(), grad_clip)

            self.optimizer.step()
            self.step += 1

            if self.step % i_log == 0 or self.step <= 10:
                # All ranks participate in the gather; rank 0 reduces + logs.
                gathered = all_gather_data(
                    {"loss": accum_loss, "metrics": accum_metrics})
                if self.is_master:
                    global_loss = float(np.mean([g["loss"] for g in gathered]))
                    global_metrics = dict_reduce(
                        [g["metrics"] for g in gathered],
                        lambda x: float(np.mean(x)),
                    )
                    self._log(global_loss, global_metrics, t0, i_log)
                t0 = time.time()
            if self.step % i_save == 0:
                self.save_checkpoint(self.step)

        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------
    # Debug / profiling training loop.
    # ------------------------------------------------------------------
    def run_training_debug(self, n_steps: int = 10):
        """Profiled clone of :meth:`run_training` that runs only ``n_steps``.

        Wraps the loop in ``torch.profiler.profile`` (CPU + CUDA) and annotates
        each phase (data / fwd / bwd / clip / opt) with ``record_function`` so
        the profiler table groups time by phase. Per-method time and memory
        hotspots are printed and tee'd to ``<exp_dir>/debug_profile/profile_log.txt``.
        A Chrome/Perfetto trace is written to ``<exp_dir>/debug_profile/trace.json``.

        Only rank 0 profiles and writes artifacts; other ranks run the loop
        plainly so collective ops (FSDP all-gather / reduce-scatter) still match.
        """
        from contextlib import nullcontext

        assert self.pipe is not None, "init_models_and_more() must run first"
        assert self.dataloader is not None, "prepare_dataloader() must run first"
        self.pipe.train()

        out_dir = os.path.join(self.exp_dir, "debug_profile")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(out_dir, f"{ts}_profile_log.txt")
        trace_path = os.path.join(out_dir, f"{ts}_trace.json")
        log_fh = None
        if self.is_master:
            os.makedirs(out_dir, exist_ok=True)
            log_fh = open(log_path, "w")

        def _tee(*parts):
            """print + training log via print_and_log, AND append to the
            profile log (rank 0 only)."""
            msg = " ".join(str(p) for p in parts)
            self.print_and_log(msg)
            if log_fh is not None:
                log_fh.write(msg + "\n")
                log_fh.flush()

        self._log_memory_footprint("pre-train")

        data_iter = iter(self.dataloader)
        ga = self.cfg.gradient_accumulation_steps
        grad_clip = self.cfg.grad_clip

        def _now():
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            return time.time()

        # Profiler schedule: skip the first step (wait), warm up two steps
        # (compile / cudnn autotune / FSDP first-iter overhead), then actively
        # record a small number of steps. Keep ``active`` capped (and avoid
        # ``with_stack``) so the exported chrome trace stays small — one or two
        # steps are enough to see the per-op breakdown, while recording every
        # step balloons trace.json to hundreds of MB.
        if self.is_master:
            wait = 1 if n_steps > 4 else 0
            warmup = 2 if n_steps > 4 else 0
            active = min(2, max(1, n_steps - wait - warmup))

            def _on_trace_ready(prof):
                prof.export_chrome_trace(trace_path)

            profiler_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=_on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        else:
            profiler_ctx = nullcontext()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        _tee(f"[debug] profiling {n_steps} steps "
             f"(grad_accum={ga}) -> {out_dir}")

        rf = torch.profiler.record_function

        with profiler_ctx as prof:
            for dbg_step in range(n_steps):
                step_t_start = _now()
                t_data = t_fwd = t_bwd = 0.0

                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                for _ in range(ga):
                    _t = _now()
                    with rf("data"):
                        batch, data_iter = self._next_batch(data_iter)
                        batch = move_to_device_and_cast(
                            batch, self.device, dtype=self.training_dtype)
                    t_data += _now() - _t

                    _t = _now()
                    with rf("forward"):
                        loss, metrics = self.compute_loss(batch)
                    t_fwd += _now() - _t

                    _t = _now()
                    with rf("backward"):
                        (loss / ga).backward()
                    t_bwd += _now() - _t

                    accum_loss += loss.detach().item()
                accum_loss /= ga

                _t = _now()
                with rf("grad_clip"):
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.pipe.parameters(), grad_clip)
                t_clip = _now() - _t

                _t = _now()
                with rf("optimizer_step"):
                    self.optimizer.step()
                t_opt = _now() - _t
                self.step += 1

                t_total = _now() - step_t_start
                if self.is_master:
                    _tee(
                        f"[timing] step {dbg_step + 1}/{n_steps} "
                        f"total={t_total * 1000:.1f}ms | "
                        f"data={t_data * 1000:.1f}ms fwd={t_fwd * 1000:.1f}ms "
                        f"bwd={t_bwd * 1000:.1f}ms clip={t_clip * 1000:.1f}ms "
                        f"opt={t_opt * 1000:.1f}ms loss={accum_loss:.4f}"
                    )

                if prof is not None:
                    prof.step()

        # ---- Report -------------------------------------------------------
        if self.is_master and prof is not None:
            ka = prof.key_averages()
            _tee("\n" + "=" * 80)
            _tee("[profile] Top ops by total CUDA time")
            _tee("=" * 80)
            _tee(ka.table(sort_by="cuda_time_total", row_limit=30))

            _tee("\n" + "=" * 80)
            _tee("[profile] Top ops by total CPU time")
            _tee("=" * 80)
            _tee(ka.table(sort_by="cpu_time_total", row_limit=30))

            _tee("\n" + "=" * 80)
            _tee("[profile] Memory hotspots (by self CUDA memory usage)")
            _tee("=" * 80)
            try:
                _tee(ka.table(sort_by="self_cuda_memory_usage", row_limit=30))
            except Exception as e:  # noqa: BLE001
                _tee(f"[profile] memory table unavailable: {e}")

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.max_memory_reserved(self.device) / 1024**3
                _tee(
                    f"\n[profile] peak CUDA memory: "
                    f"max_allocated={peak:.2f}GiB max_reserved={reserved:.2f}GiB")

            _tee(f"\n[profile] chrome trace written to {trace_path}")
            _tee(f"[profile] full log written to {log_path}")

        if log_fh is not None:
            log_fh.close()

    def fit(self):
        self.init_training_logger()
        self.init_models_and_more()
        self.prepare_dataloader()
        self.init_logger()
        self.run_training()

    def fit_debug(self, n_steps: int = 10):
        """Like :meth:`fit` but runs the profiled debug loop for ``n_steps``."""
        self.init_training_logger()
        self.init_models_and_more()
        self.prepare_dataloader()
        self.init_logger()
        self.run_training_debug(n_steps=n_steps)