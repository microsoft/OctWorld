import torch
import easydict as edict
import numpy as np
import utils3d
import torch
import trimesh
ctx = utils3d.torch.RastContext(backend='cuda')


def _rasterize_face_chunk(verts, colors, faces_chunk, resolution,
                          perspective, views, near, far, remap: bool = True):
    """Rasterize a subset of faces.

    When ``remap`` is True (default), vertices are subset to only those
    referenced by the chunk — necessary for chunked rendering to keep
    nvdiffrast within its per-call vertex/subtriangle limits and to reduce
    transform cost. When ``remap`` is False, the full vertex/color buffers
    are passed through and ``faces_chunk`` is used directly (must already be
    int32) — used by the single-chunk fast path to skip the redundant
    ``unique`` + ``index_select`` pass.
    """
    if remap:
        used_idx, remapped = torch.unique(faces_chunk.view(-1), return_inverse=True)
        sub_verts = verts.index_select(0, used_idx)
        sub_colors = colors.index_select(0, used_idx)
        sub_faces = remapped.view_as(faces_chunk).to(torch.int32).contiguous()
    else:
        sub_verts = verts
        sub_colors = colors
        sub_faces = faces_chunk if faces_chunk.dtype == torch.int32 \
            else faces_chunk.to(torch.int32)
    buffers = utils3d.torch.rasterize_triangles(
        ctx,
        resolution[1],
        resolution[0],
        vertices=sub_verts[None, :, :],
        faces=sub_faces,
        attr=sub_colors[None, :, :],
        projection=perspective,
        view=views,
        antialiasing=False,
    )
    image, depth_buf = buffers['image'][0], buffers['depth'][0]
    depth_mask = depth_buf < 1.0
    depth = utils3d.torch.depth_buffer_to_linear(depth_buf, near=near, far=far)
    depth[~depth_mask] = 0.0
    return image, depth.unsqueeze(0)  # (3, H, W), (1, H, W)


def render_mesh(mesh: trimesh.Trimesh, resolution, extrinsics, intrinsics,
                near=1e-4, far=100, max_num_faces: int = int(1e7)):
    """Rasterize a Trimesh. When the face count exceeds ``max_num_faces`` the
    mesh is split into face-chunks (vertices subset per chunk) and composited
    by nearest valid depth — analogous to ``render_point_cloud``.
    """
    device = 'cuda'
    verts = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).to(device, dtype=torch.int64)
    if not isinstance(extrinsics, torch.Tensor):
        extrinsics = torch.from_numpy(extrinsics)
    extrinsics = extrinsics.float().to(device)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.from_numpy(intrinsics)
    intrinsics = intrinsics.float().to(device)
    colors = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).to(device).float() / 255.0
    perspective = utils3d.torch.intrinsics_to_perspective(intrinsics.cpu(), near=near, far=far).to(device)
    views = utils3d.torch.extrinsics_to_view(extrinsics)

    num_faces = faces.shape[0]
    if num_faces <= max_num_faces:
        # Fast path: no chunking, pass the full vertex/color/face buffers
        # straight through to nvdiffrast — skip the unique/index_select remap.
        image, depth = _rasterize_face_chunk(
            verts, colors, faces, resolution, perspective, views, near, far,
            remap=False,
        )
        return {"color": image, "depth": depth}

    num_chunks = (num_faces + max_num_faces - 1) // max_num_faces
    render_image = None
    render_depth = None
    render_depth_cmp = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * max_num_faces
        end = min(start + max_num_faces, num_faces)
        chunk_image, chunk_depth = _rasterize_face_chunk(
            verts, colors, faces[start:end],
            resolution, perspective, views, near, far,
            remap=True,
        )
        chunk_depth_cmp = chunk_depth.clone()
        chunk_depth_cmp[chunk_depth_cmp <= 1e-3] = 1e9
        if render_image is None:
            render_image = chunk_image
            render_depth = chunk_depth
            render_depth_cmp = chunk_depth_cmp
        else:
            closer = chunk_depth_cmp < render_depth_cmp  # (1, H, W)
            closer_c = closer.expand_as(render_image)
            render_image[closer_c] = chunk_image[closer_c]
            render_depth[closer] = chunk_depth[closer]
            render_depth_cmp[closer] = chunk_depth_cmp[closer]
    # Post-process: pixels that were invalid in every chunk still carry the
    # 1e9 sentinel in render_depth_cmp; force their depth back to 0 so the
    # output matches the single-chunk convention (invalid == 0).
    render_depth[render_depth_cmp >= 1e9] = 0.0
    return {"color": render_image, "depth": render_depth}