import torch
import triton
import triton.language as tl
import numpy as np

# 这里需要预先定义好标准的 Marching Cubes Table
# edge_to_vertices: (12, 2) 定义了12条边分别连接哪两个顶点(0-7)
# tri_table: (256, 16) 定义了256种配置下对应的三角形边索引，-1结束
from .tri_table import (
    prepare_edge_to_vertex,
    prepare_face_table_tensor,
    prepare_cube_offset,
)


def _build_morton_ordered_tables():
    """Bake the std-corner -> morton-position permutation into the MC LUTs.

    Upstream the octree stores per-cube corner data in morton order:
        morton(v) = (x<<2) | (y<<1) | z
    while the canonical Bourke/Lewiner FACE_TABLE / EDGE_TO_VERTICES use a
    different corner labelling (the "std" order produced by
    ``prepare_cube_offset()``). Previously ``extract_mesh_from_octree``
    physically reshuffled the per-cube (N, 8, C) features/positions buffers
    on every call to convert morton -> std (≈3 GB transient for N≈12M).

    Permuting the kernel's LUTs once at import lets the kernel read the
    morton-ordered buffers directly, producing identical output without the
    reshuffle.

    Returns:
        edge_to_v_morton: (12, 2) int32 -- each endpoint is a morton position.
        face_table_morton: (256, 16) int32 -- row m = the entry the std table
            would return for a configuration whose corner-set is the morton
            interpretation of bit pattern m.
    """
    # std corner k lives at morton position std_to_morton[k] in the input buffer.
    cube_offset_std = prepare_cube_offset()                       # (8, 3) std order
    std_to_morton = (
        (cube_offset_std[:, 0] << 2)
        | (cube_offset_std[:, 1] << 1)
        | cube_offset_std[:, 2]
    ).tolist()                                                    # len 8 permutation
    assert sorted(std_to_morton) == list(range(8)), std_to_morton

    # 1) Edge endpoints: rewrite std-corner indices to their morton positions.
    edge_to_v_std = prepare_edge_to_vertex()                      # (12, 2) std-corner ids
    edge_to_v_morton = torch.tensor(
        [[std_to_morton[int(e0)], std_to_morton[int(e1)]]
         for e0, e1 in edge_to_v_std.tolist()],
        dtype=torch.int32,
    )

    # 2) Face table rows: re-index by the morton cube_index.
    #    cube_index_std bit k <=> (sdf at std corner k below iso)
    #                          <=> (sdf at morton position std_to_morton[k] below iso)
    #                          <=> cube_index_morton bit std_to_morton[k]
    face_table_std = prepare_face_table_tensor()                  # (256, 16)
    perm = torch.empty(256, dtype=torch.long)
    for morton_idx in range(256):
        std_idx = 0
        for k in range(8):
            if (morton_idx >> std_to_morton[k]) & 1:
                std_idx |= (1 << k)
        perm[morton_idx] = std_idx
    face_table_morton = face_table_std[perm].contiguous()

    return edge_to_v_morton, face_table_morton


_EDGE_TO_V_MORTON, _FACE_TABLE_MORTON = _build_morton_ordered_tables()
# 辅助函数：将 Table 移到 GPU
# NOTE: tables below are permuted so the kernel can consume (N, 8, ...) buffers
# in morton order directly (no runtime corner shuffle needed).
EDGE_TO_VERTICES_TENSOR = _EDGE_TO_V_MORTON.to('cuda', dtype=torch.int32)
TRI_TABLE_TENSOR = _FACE_TABLE_MORTON.to('cuda', dtype=torch.int32)

@triton.jit
def marching_cubes_triton_kernel(
    verts_ptr,          # (N, 8, 3)
    sdf_ptr,            # (N, 8)
    color_ptr,        # (N, 8, 3) 可选的属性值
    tri_table_ptr,      # (256, 16)
    edge_to_v_ptr,      # (12, 2)
    out_verts_ptr,      # (N, 15, 3) - 每个Cube最多5个三角=15个顶点
    out_colors_ptr,    # (N, 15, 3) - 每个Cube最多5个三角=15个顶点
    out_mask_ptr,       # (N, 15)    - 标记哪些顶点是有效的
    isolevel,
    N,
    stride_v_n, stride_v_v, stride_v_c,
    stride_s_n, stride_s_v,
    stride_c_n, stride_c_v, stride_c_c,
    stride_out_n, stride_out_v, stride_out_c
):
    # 获取当前处理的 Cube ID
    idx = tl.program_id(0)
    if idx >= N:
        return

    # 1. 计算 Cube Index (8-bit configuration)
    cube_index = 0
    # 这里的循环次数很小(8)，Triton会unroll它
    for i in range(8):
        s_val = tl.load(sdf_ptr + idx * stride_s_n + i * stride_s_v)
        if s_val < isolevel:
            cube_index |= (1 << i)

    # 2. 从 TriTable 获取边索引
    # tri_table 每一行有16个元素
    table_row_ptr = tri_table_ptr + cube_index * 16
    
    # 3. 遍历三角形边 (最多15条边)
    for i in range(15):
        edge_idx = tl.load(table_row_ptr + i)
        
        if edge_idx == -1:
            # 标记后续顶点无效并退出
            tl.store(out_mask_ptr + idx * 15 + i, 0)
        else:
            # 先标记该顶点有效（后面可能因为退化三角形而改为无效）
            tl.store(out_mask_ptr + idx * 15 + i, 1)

            # 4. 获取边对应的两个端点索引
            v1_idx = tl.load(edge_to_v_ptr + edge_idx * 2 + 0)
            v2_idx = tl.load(edge_to_v_ptr + edge_idx * 2 + 1)

            # 5. 加载两个端点的SDF值
            s1 = tl.load(sdf_ptr + idx * stride_s_n + v1_idx * stride_s_v)
            s2 = tl.load(sdf_ptr + idx * stride_s_n + v2_idx * stride_s_v)

            # 6. 线性插值
            # mu = (iso - s1) / (s2 - s1)
        
            if tl.abs(isolevel - s1) < 1e-6:
                mu = 0.0
            elif tl.abs(isolevel - s2) < 1e-6:
                mu = 1.0
            elif tl.abs(s2 - s1) < 1e-6:
                mu = 0.5
            else:
                mu = (isolevel - s1) / (s2 - s1)
            
            out_v_base = out_verts_ptr + idx * stride_out_n + i * stride_out_v
            out_c_base = out_colors_ptr + idx * stride_out_n + i * stride_out_v
            for j in range(3):
                p1_j = tl.load(verts_ptr + idx * stride_v_n + (v1_idx * stride_v_v) + j)
                p2_j = tl.load(verts_ptr + idx * stride_v_n + (v2_idx * stride_v_v) + j)
                res_j = p1_j + mu * (p2_j - p1_j)
                tl.store(out_v_base + j, res_j)
                c1_j = tl.load(color_ptr + idx * stride_c_n + (v1_idx * stride_c_v) + j)
                c2_j = tl.load(color_ptr + idx * stride_c_n + (v2_idx * stride_c_v) + j)
                res_cj = c1_j + mu * (c2_j - c1_j)
                tl.store(out_c_base + j, res_cj)
            
            # 7. Degenerate Triangle Filtering (内联实现)
            # 每处理完3个顶点（一个三角形），检查是否为退化三角形
            if (i + 1) % 3 == 0:
                tri_start_idx = i - 2  # 当前三角形的起始顶点索引
                EPS_DEGEN = 1e-12
                
                # 加载三个顶点的坐标基地址
                v0_base = out_verts_ptr + idx * stride_out_n + tri_start_idx * stride_out_v
                v1_base = out_verts_ptr + idx * stride_out_n + (tri_start_idx + 1) * stride_out_v
                v2_base = out_verts_ptr + idx * stride_out_n + (tri_start_idx + 2) * stride_out_v
                
                # 计算顶点之间的距离平方
                # v0 vs v1
                dist01_sq = 0.0
                for k in range(3):
                    p0_k = tl.load(v0_base + k)
                    p1_k = tl.load(v1_base + k)
                    diff = p0_k - p1_k
                    dist01_sq += diff * diff
                
                # v1 vs v2
                dist12_sq = 0.0
                for k in range(3):
                    p1_k = tl.load(v1_base + k)
                    p2_k = tl.load(v2_base + k)
                    diff = p1_k - p2_k
                    dist12_sq += diff * diff
                
                # v2 vs v0
                dist20_sq = 0.0
                for k in range(3):
                    p2_k = tl.load(v2_base + k)
                    p0_k = tl.load(v0_base + k)
                    diff = p2_k - p0_k
                    dist20_sq += diff * diff
                
                # 如果任意两点距离过近，则为退化三角形
                is_degen = (dist01_sq < EPS_DEGEN) | (dist12_sq < EPS_DEGEN) | (dist20_sq < EPS_DEGEN)
                
                if is_degen:
                    # 将该三角形的3个顶点都标记为无效
                    tl.store(out_mask_ptr + idx * 15 + tri_start_idx, 0)
                    tl.store(out_mask_ptr + idx * 15 + tri_start_idx + 1, 0)
                    tl.store(out_mask_ptr + idx * 15 + tri_start_idx + 2, 0)


def marching_cubes_triton(verts, sdf, colors, isolevel=0.0, return_mask=False):
    """Single-batch marching cubes; returns triangle-soup vertices/colors.

    Chunking and the triangle-soup -> indexed-mesh ``torch.unique`` step
    previously bundled in this module have been moved to the caller (see
    ``extract_mesh_from_octree``) so that per-chunk output buffers can be
    freed between iterations and the global unique-ify only runs once.

    Args:
        verts:   (N, 8, 3) cube corner positions, float32.
        sdf:     (N, 8) SDF values at cube corners, float32.
        colors:  (N, 8, 3) per-corner RGB, float32.
        isolevel: marching-cubes iso threshold.
        return_mask: if True, also returns the (N, 15) int32 corner-validity
            mask emitted by the kernel.

    Returns:
        ``(verts_soup, colors_soup)`` of shape (M, 3) -- triangle soup; M is
        a multiple of 3 (3 verts per generated triangle, in cube order).
        If ``return_mask`` is True, additionally returns ``out_mask`` of
        shape (N, 15) int32.
    """
    N = verts.shape[0]
    device = verts.device

    out_verts = torch.empty((N, 15, 3), device=device, dtype=torch.float32)
    out_colors = torch.empty((N, 15, 3), device=device, dtype=torch.float32)
    out_mask = torch.zeros((N, 15), device=device, dtype=torch.int32)

    grid = (N,)
    marching_cubes_triton_kernel[grid](
        verts, sdf, colors,
        TRI_TABLE_TENSOR, EDGE_TO_VERTICES_TENSOR,
        out_verts, out_colors, out_mask,
        isolevel, N,
        verts.stride(0), verts.stride(1), verts.stride(2),
        sdf.stride(0), sdf.stride(1),
        colors.stride(0), colors.stride(1), colors.stride(2),
        out_verts.stride(0), out_verts.stride(1), out_verts.stride(2),
        num_warps=4,
    )

    valid_mask = out_mask.bool()                # (N, 15)
    verts_soup = out_verts[valid_mask]          # (M, 3)
    colors_soup = out_colors[valid_mask]        # (M, 3)

    if return_mask:
        return verts_soup, colors_soup, out_mask
    return verts_soup, colors_soup