# Adapted from pytorch/examples:
# https://github.com/pytorch/examples/blob/main/distributed/FSDP2/checkpoint.py
# Local changes: hardcode the DCP state-dict API path, drop the
# dcp_api/dtensor_api folder split, and key checkpoints by training step
# instead of wall-clock timestamp.
#
# Resume checkpoints (keyed by step) are stored as *sharded* DCP directories:
# each rank reads/writes only its own shard, so save/load never gathers the
# full model onto a single rank. Init checkpoints (a single full .pt supplied
# via ``init_ckpt_path``) still go through the full-state broadcast path.
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


# Retained for backward-compatible imports (e.g. validate.py). The sharded DCP
# format writes a directory of shards, not these single-file names.
MODEL_CHECKPOINT = "model_state_dict.pt"

# The (epoch, idx) sampler state is identical across ranks (all ranks advance in
# lockstep), so it is written once by rank 0 as a small file inside the step dir
# alongside the sharded DCP model/optim shards.
SAMPLER_CHECKPOINT = "sampler_state.pt"

# Completion marker written by rank 0 *after* every rank has finished writing its
# DCP shard (and the sampler file). A step dir without this marker is treated as
# incomplete (e.g. the job was preempted mid-save) and is skipped on resume, so
# resume never picks a half-written checkpoint.
SUCCESS_MARKER = "SUCCESS.txt"


def get_latest_checkpoint_folder(path, require_success: bool = False):
    """Return the highest numeric step-dir under ``path`` (or ``None``).

    Args:
        require_success: when True, only consider step dirs that contain a
            :data:`SUCCESS_MARKER` file (i.e. a fully-completed save). Used on
            resume so a preempted, half-written checkpoint is ignored. Defaults
            to False for backward-compatible callers (e.g. validate.py) that
            want the latest dir regardless of the marker.
    """
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
            except ValueError:
                continue  # Skip non-numeric folder names
            if require_success and not os.path.exists(
                    os.path.join(folder_path, SUCCESS_MARKER)):
                continue  # Skip incomplete (preempted mid-save) checkpoints
            if max_num is None or num > max_num:
                max_num = num
    return max_num


class Checkpointer:
    def __init__(self, folder: str):
        self.folder = folder
        # Resume only from a fully-completed save (SUCCESS.txt present), so a
        # checkpoint that was preempted mid-write is never picked.
        self.last_step = get_latest_checkpoint_folder(folder, require_success=True)

    def is_empty(self):
        return self.last_step is None

    @property
    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def _checkpoint_id(self, step: int) -> str:
        return f"{self.folder}/{step}"

    @staticmethod
    def _materialize_meta_(model: torch.nn.Module, device: torch.device):
        # Sharded DCP load reads each rank's shard into pre-allocated storage,
        # so any meta params/buffers (from meta-init + apply_fsdp) must first be
        # given real (empty) storage. Non-persistent buffers not in the
        # checkpoint (e.g. RoPE freqs) are rebuilt afterwards by pipe.materialize().
        if any(t.is_meta for t in model.parameters()) or any(
            t.is_meta for t in model.buffers()
        ):
            model.to_empty(device=device)

    def load_model(self, model: torch.nn.Module):
        self._materialize_meta_(model, self._device)
        state_dict = {"model": get_model_state_dict(model)}
        dcp.load(state_dict, checkpoint_id=self._checkpoint_id(self.last_step))
        set_model_state_dict(model=model, model_state_dict=state_dict["model"])

    def load_model_from_path(self, model: torch.nn.Module, path: str):
        # Init checkpoints are a single full-state .pt (not a sharded DCP dir):
        # load on every rank and let DCP reshard via broadcast_from_rank0.
        full_sd = torch.load(
            path, mmap=True, weights_only=True, map_location="cpu"
        )
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

    def load_optim(self, model: torch.nn.Module, opt: torch.optim.Optimizer):
        state_dict = {"optim": get_optimizer_state_dict(model, optimizers=opt)}
        dcp.load(state_dict, checkpoint_id=self._checkpoint_id(self.last_step))
        set_optimizer_state_dict(
            model=model,
            optimizers=opt,
            optim_state_dict=state_dict["optim"],
        )

    def _sampler_path(self, step: int) -> str:
        return os.path.join(self._checkpoint_id(step), SAMPLER_CHECKPOINT)

    def _success_path(self, step: int) -> str:
        return os.path.join(self._checkpoint_id(step), SUCCESS_MARKER)

    def load_sampler(self, sampler) -> bool:
        # Sampler state is a single small file shared by all ranks. Older
        # checkpoints predating sampler resumability won't have it; skip in that
        # case so loading stays backward compatible.
        path = self._sampler_path(self.last_step)
        if not os.path.exists(path):
            return False
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        sampler.load_state_dict(state_dict)
        return True

    def save(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        step: int,
        sampler=None,
    ):
        # DCP save is collective: every rank writes its own shard. Do NOT guard
        # this with a rank-0 check.
        state_dict = {
            "model": get_model_state_dict(model),
            "optim": get_optimizer_state_dict(model, optimizers=optim),
        }
        dcp.save(state_dict, checkpoint_id=self._checkpoint_id(step))
        # The sampler state is identical on every rank, so only rank 0 writes it
        # (into the step dir dcp.save just created).
        is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        if sampler is not None and is_main:
            torch.save(sampler.state_dict(), self._sampler_path(step))
        # Barrier so rank 0 only stamps SUCCESS.txt once EVERY rank has finished
        # writing its DCP shard (dcp.save returned) and the sampler file is on
        # disk. If the job is preempted before this marker exists, the step dir
        # is treated as incomplete and skipped on resume.
        if dist.is_initialized():
            dist.barrier()
        if is_main:
            with open(self._success_path(step), "w") as f:
                f.write("ok\n")
        self.last_step = step
