from ..octree.dynamic_octree import DynOctree
from .marching_cube_triton import marching_cubes_triton
import torch
import trimesh
import numpy as np
import time
from pathlib import Path
from typing import List, Optional


def _build_cube_summary_list(mesh_cube_pts: torch.Tensor, mesh_cube_features: torch.Tensor):
    if mesh_cube_pts.shape[0] == 0:
        return []

    cube_centers = mesh_cube_pts.mean(dim=1)
    cube_extents = mesh_cube_pts.max(dim=1).values - mesh_cube_pts.min(dim=1).values
    cube_edge_lengths = cube_extents.mean(dim=1)
    cube_mean_colors = mesh_cube_features[:, :, 1:4].mean(dim=1)

    return [
        {
            "center": center.tolist(),
            "edge_length": float(edge_length.item()),
            "mean_color": mean_color.tolist(),
        }
        for center, edge_length, mean_color in zip(cube_centers, cube_edge_lengths, cube_mean_colors)
    ]


def _dump_cube_summary_list(cube_summary_list, dump_path):
    dump_path = Path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("w") as f:
        for cube_summary in cube_summary_list:
            f.write(f"{cube_summary}\n")

def extract_mesh_from_octree(
    octree : DynOctree,
    isolevel: float = 0.0,
    return_mask: bool = False,
    dump_cube_summary_list: bool = False,
    cube_summary_dump_path: str = "debugs/cube_summary_list.txt",
    per_depth_indices: Optional[List[torch.Tensor]] = None,
    weight_threshold: float = 1.0,
    verbose: bool = False,
):
    """
    从八叉树中提取网格，使用 Marching Cubes 算法。

    Args:
        octree: FlexOctree 对象，包含八叉树数据结构。
        isolevel: 用于确定等值面的阈值。
        per_depth_indices: 可选，每层八叉树的节点索引列表。如果提供，则只对这些
            节点（filter 到 leaf 之后）做 marching cubes，跳过全场景 fetch；典型
            用法是先用 ``DynOctree.fetch_active_nodes_by_frustum`` 做视锥剔除拿到
            一个 batch 的 view 内节点，再调用本函数做 view-based mesh extraction。
        weight_threshold: 当 ``per_depth_indices is None`` 时，全场景模式下使用
            的 weight 过滤阈值。
    Returns:
        vertices: 提取的网格顶点列表。
        faces: 提取的网格面列表。
    """
    if per_depth_indices is None:
        features, weights, pts, _ = octree.fetch_active_points(weight_threshold=weight_threshold)
    else:
        features, weights, pts, _ = octree.fetch_points_by_indices(
            per_depth_indices, leaf_only=True,
        )

    _use_mem = verbose and torch.cuda.is_available()
    stage_stats: dict = {}

    def _begin():
        if _use_mem:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def _end(label: str, t0: float, extra: str = ""):
        if _use_mem:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            peak = float("nan")
        stage_stats[label] = (time.perf_counter() - t0, peak, extra)

    if features.shape[0] == 0:
        # No cubes to march; return empty tensors with the expected shapes.
        device = features.device
        empty_v = torch.zeros((0, 3), device=device, dtype=features.dtype)
        empty_f = torch.zeros((0, 3), device=device, dtype=torch.long)
        empty_c = torch.zeros((0, 3), device=device, dtype=features.dtype)
        if return_mask:
            empty_m = torch.zeros((0, 5), device=device, dtype=torch.bool)
            return empty_v, empty_f, empty_c, empty_m
        return empty_v, empty_f, empty_c

    # In-place "unobserved corner -> sdf 1" assignment. The original code did
    # ``torch.where(weights>0, features, ones_like(features))`` which
    # allocates two extra N*8*C buffers. ``masked_fill_`` does the same thing
    # in-place; ``features`` came from a fresh torch.cat in fetch_*_points,
    # so it's safe to modify.
    t0 = _begin()
    features.masked_fill_((weights == 0).unsqueeze(-1), 1.0)
    del weights  # not needed past this point

    # to cubes version
    features = features.reshape(-1, 8, features.shape[-1])
    pts = pts.reshape(-1, 8, 3)

    # NOTE: the per-cube corner buffers from the octree are natively in morton
    # order (see ``DynOctree.xyz_to_points``). The MC kernel's LUTs have been
    # baked into the same morton convention at module import (see
    # ``marching_cube_triton._build_morton_ordered_tables``), so the runtime
    # std-corner reindex that used to live here -- and allocated a ~3 GB
    # transient on chunk 3 -- is now unnecessary.
    _end("prep_morton_reindex", t0, extra=f"N={features.shape[0]}")

    # Chunked marching cubes. Previously this chunking lived inside
    # ``marching_cubes_triton``; pulling it out lets us free the per-chunk
    # output tensors before allocating the next chunk's, and keeps the
    # triangle-soup-to-indexed-mesh step (the torch.unique) as a single pass
    # at the end instead of an O(N) intermediate.
    N = features.shape[0]
    max_cubes_per_chunk = int(2e6)
    device = features.device

    verts_soup_list: List[torch.Tensor] = []
    colors_soup_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []  # only used if return_mask / dump_cube_summary_list

    need_mask = return_mask or dump_cube_summary_list

    t0 = _begin()
    for i in range(0, N, max_cubes_per_chunk):
        end = min(i + max_cubes_per_chunk, N)
        p_b = pts[i:end]
        f_b = features[i:end]
        sdf_b = f_b[:, :, 0]
        col_b = f_b[:, :, 1:4]
        if need_mask:
            v_soup, c_soup, m_b = marching_cubes_triton(
                p_b, sdf_b, col_b, isolevel=isolevel, return_mask=True,
            )
            mask_list.append(m_b)
        else:
            v_soup, c_soup = marching_cubes_triton(
                p_b, sdf_b, col_b, isolevel=isolevel, return_mask=False,
            )
        verts_soup_list.append(v_soup)
        colors_soup_list.append(c_soup)
        del p_b, f_b, sdf_b, col_b
    _end("marching_cubes_chunked", t0,
         extra=f"N={N} chunks={(N + max_cubes_per_chunk - 1)//max_cubes_per_chunk}")

    t0 = _begin()
    verts_soup = torch.cat(verts_soup_list, dim=0)
    colors_soup = torch.cat(colors_soup_list, dim=0)
    verts_soup_list.clear()
    colors_soup_list.clear()
    _end("cat_soup", t0, extra=f"M={verts_soup.shape[0]}")

    # Triangle soup -> indexed mesh (single global unique).
    t0 = _begin()
    final_verts, inverse_indices = torch.unique(verts_soup, return_inverse=True, dim=0)
    final_faces = inverse_indices.view(-1, 3)
    final_colors = torch.zeros((final_verts.shape[0], 3), device=device, dtype=torch.float32)
    final_colors.index_copy_(0, inverse_indices, colors_soup)
    del verts_soup, colors_soup
    _end("unique_index_copy", t0, extra=f"V={final_verts.shape[0]}")

    if verbose:
        print("  [extract_mesh_from_octree stages]")
        for k, (dt, peak, extra) in stage_stats.items():
            print(f"    {k:<28s} {dt*1000:8.1f} ms   peak {peak:8.1f} MB   {extra}")

    if dump_cube_summary_list:
        masks = torch.cat(mask_list, dim=0)
        cube_has_mesh_mask = masks.bool().any(dim=1)
        mesh_cube_indices = torch.nonzero(cube_has_mesh_mask, as_tuple=True)[0]
        mesh_cube_pts = pts[mesh_cube_indices]
        mesh_cube_features = features[mesh_cube_indices]
        cube_summary_list = _build_cube_summary_list(mesh_cube_pts, mesh_cube_features)
        _dump_cube_summary_list(cube_summary_list, cube_summary_dump_path)

    if return_mask:
        masks = torch.cat(mask_list, dim=0) if mask_list else torch.zeros(
            (0, 15), device=device, dtype=torch.int32,
        )
        return final_verts, final_faces, final_colors, masks.bool()
    return final_verts, final_faces, final_colors


def dense_volume_marching_cubes(volume_features: torch.Tensor):
    grid_reso = volume_features.shape[0]

    # Corner offsets in MORTON order: index m -> ((m>>2)&1, (m>>1)&1, m&1).
    # Must match the layout the (permuted) MC kernel LUTs expect; see
    # ``marching_cube_triton._build_morton_ordered_tables``.
    morton_offsets = torch.tensor(
        [((m >> 2) & 1, (m >> 1) & 1, m & 1) for m in range(8)],
        device='cuda',
    )

    # 找到每个 cube 的起始坐标 (bottom-front-left)
    base_indices = torch.stack(torch.meshgrid(
        torch.arange(grid_reso-1, device='cuda'),
        torch.arange(grid_reso-1, device='cuda'),
        torch.arange(grid_reso-1, device='cuda'),
        indexing='ij'
    ), dim=-1).reshape(-1, 3) # (N_cubes, 3)

    # 展开成 (N_cubes, 8, 3)
    points = base_indices.unsqueeze(1) + morton_offsets.unsqueeze(0).to(base_indices)
    cube_features = volume_features[points[..., 0], points[..., 1], points[..., 2]]  # (N_cubes, 8, 4)

    # --- 3. 调用 Triton Marching Cubes ---
    # marching_cubes_triton now returns triangle-soup; do the unique-ify here.
    verts_soup, colors_soup = marching_cubes_triton(
        points.float(),
        cube_features[:, :, 0],
        cube_features[:, :, 1:4],
        isolevel=0.0,
    )
    vertices, inverse_indices = torch.unique(verts_soup, return_inverse=True, dim=0)
    faces = inverse_indices.view(-1, 3)
    colors = torch.zeros((vertices.shape[0], 3), device=vertices.device, dtype=torch.float32)
    colors.index_copy_(0, inverse_indices, colors_soup)

    vertices = vertices / grid_reso * 2.0 - 1.0  # normalize to [-1, 1]

    return vertices, faces, colors
