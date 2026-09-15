"""Distribution-agnostic helpers shared by the training stack.

Pure functions extracted from :mod:`octworld.training.basic_trainer` so they can
be reused by concrete trainers (e.g. the Wan trainer) without importing the
trainer class itself.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def dtype_from_str(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(name)]


def all_gather_data(data):
    """Gather an arbitrary picklable object from every rank onto rank 0.

    Returns a list of length ``world_size`` (one entry per rank) on rank 0, or
    ``None`` on the other ranks. When not running distributed, returns
    ``[data]`` so callers can handle both cases uniformly. Must be called by
    *all* ranks (it runs an ``all_gather_object`` collective).
    """
    if not dist.is_available() or not dist.is_initialized():
        return [data]

    world_size = dist.get_world_size()
    output_objects = [None for _ in range(world_size)]
    dist.all_gather_object(output_objects, data)

    if dist.get_rank() == 0:
        return output_objects
    return None


def dict_reduce(dicts, func, special_func={}):
    """Reduce a list of (possibly nested) dicts elementwise.

    For each key the matching values across ``dicts`` are gathered and passed
    to ``func`` (or ``special_func[key]`` if provided). Nested dicts are
    reduced recursively. Keys missing from some dicts are simply skipped for
    that dict (so empty time-bins don't break the reduction).
    """
    if not dicts:
        return {}
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for k in keys:
        values = [d[k] for d in dicts if k in d]
        if isinstance(values[0], dict):
            result[k] = dict_reduce(values, func, special_func)
        else:
            result[k] = special_func.get(k, func)(values)
    return result


def dict_flatten(d, parent_key="", sep="/"):
    """Flatten a nested dict, joining keys with ``sep`` (e.g. ``bin_0/mse``)."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(dict_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def metrics_to_python(value):
    """Recursively detach tensors to python scalars inside a metric dict."""
    if torch.is_tensor(value):
        return value.detach().item()
    if isinstance(value, dict):
        return {k: metrics_to_python(v) for k, v in value.items()}
    return value


def move_to_device_and_cast(value, device, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device, dtype=dtype, non_blocking=True)
    if isinstance(value, list):
        return [move_to_device_and_cast(v, device, dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device_and_cast(v, device, dtype) for v in value)
    if isinstance(value, dict):
        return {k: move_to_device_and_cast(v, device, dtype) for k, v in value.items()}
    return value


def build_mesh(world_size: int, local_world_size: int, mesh_mode: str):
    """Build DeviceMesh.

    Args:
        world_size: Total number of distributed ranks.
        local_world_size: Number of ranks per node.
        mesh_mode: "1d" for full sharding across all GPUs, or "2d" for
            HSDP (shard intra-node, replicate inter-node).
    """
    if mesh_mode == "1d":
        mesh = init_device_mesh("cuda", (world_size,))
        print(f"Using 1D FSDP mesh: {world_size} devices fully sharded")
    elif mesh_mode == "2d":
        num_nodes = world_size // local_world_size
        if num_nodes <= 1:
            mesh = init_device_mesh("cuda", (world_size,))
            print(
                f"Using 1D FSDP mesh (single node): {world_size} devices")
        else:
            mesh = init_device_mesh(
                "cuda",
                (num_nodes, local_world_size),
                mesh_dim_names=("replicate", "shard"),
            )
            print(
                f"Using 2D HSDP mesh: {num_nodes} nodes × "
                f"{local_world_size} GPUs/node "
                f"(shard intra-node, replicate inter-node)"
            )
    else:
        raise ValueError(
            f"Unknown mesh_mode={mesh_mode!r}, expected '1d' or '2d'")
    return mesh


def time_binned_mse(pred: torch.Tensor, target: torch.Tensor,
                    t: torch.Tensor, num_bins: int = 10) -> dict:
    """Per-instance MSE bucketed by timestep ``t`` in ``[0, 1]``.

    Returns a nested dict ``{f'bin_{i}': {'mse': <mean>}}`` for non-empty
    bins, matching the diffusion-loss logging convention. ``pred`` and
    ``target`` are ``[N, ...]`` tensors; ``t`` is ``[N]``.
    """
    mse_per_instance = np.array([
        torch.nn.functional.mse_loss(pred[i], target[i]).item()
        for i in range(pred.shape[0])
    ])
    time_bin = np.digitize(
        t.detach().cpu().numpy(), np.linspace(0, 1, num_bins + 1)) - 1
    terms = {}
    for i in range(num_bins):
        mask = time_bin == i
        if mask.sum() != 0:
            terms[f"bin_{i}"] = {"mse": float(mse_per_instance[mask].mean())}
    return terms
