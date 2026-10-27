import torch
import easydict as edict
import numpy as np
import utils3d
import trimesh
ctx = utils3d.torch.RastContext(backend='cuda')


def _render_point_cloud_chunk(xyz, rgb, resolution, extrinsics: torch.Tensor, intrinsics: torch.Tensor,
    near=1e-4, far=100):
    """Render a single chunk of point cloud. Internal function."""
    view = utils3d.extrinsics_to_view(extrinsics)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    render_height, render_width = resolution
    perspective = utils3d.torch.intrinsics_to_perspective(intrinsics, near=near, far=far)
    square_vertices, square_faces = utils3d.numpy.create_square_mesh(tri=True)
    device = 'cuda'
    points = xyz.to(device).float()
    colors = torch.cat([rgb, torch.ones((len(points), 1), device=points.device)], dim=-1).to(device).float()
    square_vertices = torch.from_numpy(square_vertices).to(device)
    square_faces = torch.from_numpy(square_faces).to(device)
    
    points = points @ view[:3, :3].T + view[:3, 3]
    points_z = points[:, 2].abs()
    # based on points z mean to decide the size
    scale_factor = 2.5 / (fx * render_width + fy * render_height)
    sizes = points_z * scale_factor
    sizes = sizes.to(device).float()
    vertices = (points[:, None, :] + square_vertices * sizes[:, None, None]).reshape((1, -1, 3))
    colors = colors[:, None, :].repeat(1, 4, 1).reshape((1, -1, colors.shape[-1]))
    faces = (square_faces + torch.arange(len(points), dtype=torch.int32, device='cuda')[:, None, None] * 4).reshape(-1, 3)
    buffers = utils3d.torch.rasterize_triangles(
        ctx,
        render_width,
        render_height,
        vertices=vertices,
        faces=faces,
        attr=colors,
        projection=perspective,
        antialiasing=False
    )
    render_image, render_depth = buffers['image'][0], buffers['depth'][0]
    depth_mask = render_depth < 1.0
    render_depth = utils3d.torch.depth_buffer_to_linear(render_depth, near=near, far=far)
    render_depth[~depth_mask] = 0.0
    render_image = render_image[:3]
    return {
        "color": render_image,
        "depth": render_depth.unsqueeze(0)  # (1, H, W)
    }


def render_point_cloud(xyz, rgb, resolution, extrinsics: torch.Tensor, intrinsics: torch.Tensor, max_num_points: int = int(7e6)):
    num_points = xyz.shape[0]
    if num_points <= max_num_points:
        return _render_point_cloud_chunk(xyz, rgb, resolution, extrinsics, intrinsics)
    
    # Split into chunks and composite by depth
    num_chunks = (num_points + max_num_points - 1) // max_num_points
    render_image = None
    render_depth = None
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * max_num_points
        end_idx = min((chunk_idx + 1) * max_num_points, num_points)
        chunk_result = _render_point_cloud_chunk(
            xyz[start_idx:end_idx],
            rgb[start_idx:end_idx],
            resolution, extrinsics, intrinsics
        )
        chunk_color = chunk_result["color"]   # (3, H, W)
        chunk_depth = chunk_result["depth"]   # (1, H, W)
        # Mark invalid depth as very large so it loses in comparison
        chunk_depth_cmp = chunk_depth.clone()
        chunk_depth_cmp[chunk_depth_cmp <= 1e-3] = 1e9
        
        if render_image is None:
            render_image = chunk_color
            render_depth = chunk_depth
            render_depth_cmp = chunk_depth_cmp
        else:
            # Keep pixels with closer depth
            closer_mask = chunk_depth_cmp < render_depth_cmp  # (1, H, W)
            closer_mask_color = closer_mask.expand_as(render_image)  # (3, H, W)
            render_image[closer_mask_color] = chunk_color[closer_mask_color]
            render_depth[closer_mask] = chunk_depth[closer_mask]
            render_depth_cmp[closer_mask] = chunk_depth_cmp[closer_mask]
    
    return {
        "color": render_image,
        "depth": render_depth  # (1, H, W)
    }
