import torch
from typing import Tuple


def build_frustum_planes(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build frustum planes for a batch of cameras.
    
    The intrinsics here follow the convention used in PinholeCamera:
    - intrinsics are normalized (fx, fy, cx, cy in [0,1] range)
    - extrinsics are world-to-camera (4x4)
    
    Since plane normals are normalized, image_width/height factors cancel out
    and only the normalized intrinsic values are needed.
    
    Args:
        intrinsics: (B, 3, 3) normalized camera intrinsics
        extrinsics: (B, 4, 4) world-to-camera extrinsics
        
    Returns:
        plane_normals: (B, 4, 3) frustum plane normals in world space
        plane_offsets: (B, 4) plane offsets (dot(normal, point) + offset >= 0 means inside)
    """
    device = intrinsics.device
    B = intrinsics.shape[0]
    
    # Normalized focal lengths and principal points
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    cx = intrinsics[:, 0, 2]  # (B,)
    cy = intrinsics[:, 1, 2]  # (B,)
    
    # Build 4 frustum plane normals in camera space (W and H cancel after normalization)
    # left:   ( fx, 0,  cx)
    # right:  (-fx, 0,  1-cx)
    # top:    (0,  fy,  cy)
    # bottom: (0, -fy,  1-cy)
    
    # (B, 4, 3)
    normals_cam = torch.zeros(B, 4, 3, device=device)
    normals_cam[:, 0, 0] = fx
    normals_cam[:, 0, 2] = cx
    normals_cam[:, 1, 0] = -fx
    normals_cam[:, 1, 2] = 1.0 - cx
    normals_cam[:, 2, 1] = fy
    normals_cam[:, 2, 2] = cy
    normals_cam[:, 3, 1] = -fy
    normals_cam[:, 3, 2] = 1.0 - cy
    
    # Normalize
    normals_cam = normals_cam / normals_cam.norm(dim=-1, keepdim=True)
    
    # Extract rotation from extrinsics (world-to-camera)
    R = extrinsics[:, :3, :3]  # (B, 3, 3)
    t = extrinsics[:, :3, 3]   # (B, 3)
    
    # Transform normals from camera space to world space
    # camera normal n_cam, world normal n_world: n_world = R^T @ n_cam
    # Because for w2c: p_cam = R @ p_world + t
    # plane equation in cam: n_cam . p_cam >= 0
    # => n_cam . (R @ p_world + t) >= 0
    # => (R^T @ n_cam) . p_world + n_cam . t >= 0
    # So world normal = R^T @ n_cam, offset = n_cam . t
    
    # (B, 4, 3) @ (B, 3, 3)^T -> need to be careful with dims
    # normals_world[b, i] = R[b]^T @ normals_cam[b, i]
    normals_world = torch.einsum('bpc,bic->bip', R, normals_cam)  # (B, 4, 3) -- R^T @ n
    # Wait, let me redo: R^T @ n_cam = (R.T) @ n_cam
    # einsum: normals_world[b,i,p] = R[b,c,p] * normals_cam[b,i,c] summed over c
    # That's R^T since R[c,p] transposed is R[p,c]
    # Actually R is (B,3,3), normals_cam is (B,4,3)
    # R^T @ n_cam: result[b,i,j] = sum_k R[b,k,j] * normals_cam[b,i,k]
    normals_world = torch.einsum('bkj,bik->bij', R, normals_cam)  # (B, 4, 3)
    
    # offset: n_cam . t per plane
    # offsets[b,i] = sum_k normals_cam[b,i,k] * t[b,k]
    offsets = torch.einsum('bik,bk->bi', normals_cam, t)  # (B, 4)
    
    return normals_world, offsets


def frustum_aabb_intersection(
    centers: torch.Tensor,
    extents: torch.Tensor,
    plane_normals: torch.Tensor,
    plane_offsets: torch.Tensor,
    max_box_plane_products: int = 20_000_000,
) -> torch.Tensor:
    """
    Test AABB boxes against multiple camera frustums in parallel.
    A box is considered "inside" if it is inside ANY of the frustums.

    Args:
        centers: (N, 3) AABB centers in world space
        extents: (N, 3) AABB half-extents (same for all 3 axes for cubes)
        plane_normals: (B, 4, 3) frustum plane normals in world space
        plane_offsets: (B, 4) plane offsets
        max_box_plane_products: Upper bound on ``chunk * B`` per iteration.
            The dominant intermediate is (chunk, B, 4) float32 (~16 B/box/frustum),
            so e.g. budget=2e7 caps it at ~320 MB regardless of how many cameras
            ``B`` the caller passes. Default 2e7 matches the previous
            (chunk_size=1M, B=20) integrate setting.

    Returns:
        inside_mask: (N,) boolean mask, True if the AABB intersects ANY frustum
    """
    N = centers.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=centers.device)

    B = plane_normals.shape[0]
    chunk_size = max(1, int(max_box_plane_products) // max(1, B))

    # Precompute |normals| once across all chunks.
    abs_normals = plane_normals.abs()  # (B, 4, 3)

    inside_any = torch.empty(N, dtype=torch.bool, device=centers.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c = centers[start:end]
        e = extents[start:end]

        # signed distance + projected radius per plane, fused so the (n,B,4)
        # intermediate is allocated just once and reused via in-place ops.
        # (n,B,4): s = c·normal + offset; r = e·|normal|; outside if s + r < 0
        sr = torch.einsum('nk,bpk->nbp', c, plane_normals)
        sr.add_(plane_offsets.unsqueeze(0))
        sr.add_(torch.einsum('nk,bpk->nbp', e, abs_normals))

        # outside this frustum if (s + r) < 0 for any plane
        outside_per_plane = sr < 0           # (n, B, 4) bool
        del sr
        outside_per_frustum = outside_per_plane.any(dim=2)  # (n, B)
        del outside_per_plane
        inside_any[start:end] = ~outside_per_frustum.all(dim=1)

    return inside_any
