from .camera import PinholeCamera
from .octree.dynamic_octree import DynOctree

import torch
from dataclasses import dataclass


class RayCast_Carver:
    def __init__(self,
        octree : DynOctree, 
        num_points_per_ray=36,
        carve_margin_ratio=2.0,
        use_carve_cube=False,
        cube_extend_size=2,
        octree_depth_strategy="pixel_size_to_volume_size",
        bigger_voxel_pixel_size_scale=2.0,
        ):
        '''
        Initialize the RayCast_Carver with the given octree.
        Args:
            octree: The octree to be carved.
            num_points_per_ray: Number of points to sample along each ray.
            carve_margin_ratio: The truncation margin ratio.
            use_carve_cube: Whether to use cube carving mode.
            octree_depth_strategy: Strategy to decide per-ray octree depth.
                "pixel_size_to_volume_size" uses octree.pixel_size_to_volume_size(pixel_sizes).
                "bigger_voxels" uses octree.pixel_size_to_volume_size(
                pixel_sizes * bigger_voxel_pixel_size_scale).
                "max_depth" uses octree.depth for all rays.
        '''
        self.octree = octree
        self.num_points_per_ray = num_points_per_ray
        self.carve_margin_ratio = carve_margin_ratio
        self.use_carve_cube = use_carve_cube
        self.cube_extend_size = cube_extend_size
        self.octree_depth_strategy = octree_depth_strategy
        self.bigger_voxel_pixel_size_scale = float(bigger_voxel_pixel_size_scale)
        self.cached_xyzd = []

        if self.octree_depth_strategy not in {"pixel_size_to_volume_size", "bigger_voxels", "max_depth"}:
            raise ValueError(
                f"Unknown octree_depth_strategy: {self.octree_depth_strategy}. "
                "Expected one of {'pixel_size_to_volume_size', 'bigger_voxels', 'max_depth'}."
            )
        if self.bigger_voxel_pixel_size_scale <= 0:
            raise ValueError(
                "bigger_voxel_pixel_size_scale must be > 0, "
                f"got {self.bigger_voxel_pixel_size_scale}."
            )
        
    def construct_neighbor_cubes(self, points : torch.Tensor, volume_sizes : torch.Tensor):
        '''
        construct_neighbor_cubes 
        points : (N_d, 3)
        volume_sizes : (N_d,)
        '''
        # construct a cube around the points
        cube_size = self.cube_extend_size * 2 + 1
        cube = torch.ones((cube_size, cube_size, cube_size), device=points.device)
        neighbor_grid = torch.nonzero(cube) - self.cube_extend_size # (cube_size ** 3,  3)
        neigh_points = points[:, None, :] + neighbor_grid[None, :, :] * volume_sizes[:, None, None]
        return neigh_points
        
        
    def register_frame(self, depth_img : torch.Tensor, camera : PinholeCamera):
        """
        Perform ray casting to carve out parts of the scene based on the ray origin and direction.
        Args:
            depth_img: (H, W) Depth map
            camera: PinholeCamera object
        Returns:
            indices: List of indices of octree nodes to be updated
            keys: List of keys of octree nodes to be updated
        """
        # Backproject depth map to point cloud and get pixel sizes
        valid_mask = depth_img > 0
        rays_pos, pixel_sizes = camera.back_project_points(depth_img, return_pixel_sizes=True)
        
        # fetch the valid rays only
        rays_pos = rays_pos[valid_mask] # (N_valid, 3)
        pixel_sizes = pixel_sizes[valid_mask] # (N_valid,)
        
        # Decide volume size / octree depth for each ray point.
        if self.octree_depth_strategy == "pixel_size_to_volume_size":
            # Scale by projected pixel size for adaptive depth selection.
            volume_sizes, octree_depths = self.octree.pixel_size_to_volume_size(pixel_sizes)
        elif self.octree_depth_strategy == "bigger_voxels":
            volume_sizes, octree_depths = self.octree.pixel_size_to_volume_size(
                pixel_sizes * self.bigger_voxel_pixel_size_scale
            )
        else:
            # Use the finest octree level for all points.
            max_depth = int(self.octree.depth)
            max_voxel_size = float(self.octree.get_voxel_size(max_depth))
            volume_sizes = torch.full_like(pixel_sizes, max_voxel_size)
            octree_depths = torch.full_like(pixel_sizes, max_depth, dtype=torch.long)

        if self.use_carve_cube:
            neigh_points = self.construct_neighbor_cubes(rays_pos, volume_sizes)
            neigh_oct_depth = octree_depths[:, None].expand_as(neigh_points[..., 0])
            valid_pts = (neigh_points > -1.0).all(dim=-1) & (neigh_points < 1.0).all(dim=-1)
            neigh_points = neigh_points[valid_pts]
            neigh_oct_depth = neigh_oct_depth[valid_pts]

        # compute ray origins and directions
        rays_o = camera.camera_origin
        rays_dir = rays_pos - rays_o[None, :]
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
        
        # sample points along the rays based on the carve truncation
        # the points is between [rays_pos - carve_trunc * volume_size * dir, ray_pos + carve_trunc * volume_size * dir]
        N_valid = rays_pos.shape[0]
        carve_trunc = self.carve_margin_ratio * volume_sizes
        near_pts = rays_pos - carve_trunc[..., None] * rays_dir
        far_pts = rays_pos + carve_trunc[..., None] * rays_dir
        t_vals = torch.linspace(0.0, 1.0, self.num_points_per_ray, device=depth_img.device)
        t_vals = t_vals[None, :, None].expand(N_valid, self.num_points_per_ray, 1)
        sample_pts = near_pts[..., None, :] * (1.0 - t_vals) + far_pts[..., None, :] * t_vals  # (N_valid, num_points_per_ray, 3)
        octree_depths = octree_depths[:, None].expand(N_valid, self.num_points_per_ray)
        valid_pts = (sample_pts > -1.0).all(dim=-1) & (sample_pts < 1.0).all(dim=-1)
        sample_pts = sample_pts[valid_pts]
        octree_depths = octree_depths[valid_pts]
        
        if self.use_carve_cube:
            sample_pts = torch.cat([sample_pts, neigh_points], dim=0)
            octree_depths = torch.cat([octree_depths, neigh_oct_depth], dim=0)  
        
        self.cached_xyzd.append(self.octree.points_to_xyzd(sample_pts, octree_depths, return_unique=True))

        # self.octree.allocate_points(sample_pts.reshape(-1, 3), octree_depths.reshape(-1), extend_neighbors=False)
    
    def run(self):
        '''
        Run the ray casting carver to allocate the octree structure.
        
        Returns:
            per_depth_indices: List[Tensor] from allocate_points, one per depth level,
                containing node indices of allocated points. Can be passed directly
                to octree.fetch_points_by_indices(). Returns None if nothing was cached.
            min_octree_depth: int, the shallowest (smallest) octree depth across all
                points registered since the last run. Returns None if nothing was cached.
        '''
        if len(self.cached_xyzd) == 0:
            return None, None
        
        xyzd = torch.cat(self.cached_xyzd, dim=0)
        xyzd = torch.unique(xyzd, dim=0)

        min_octree_depth = int(xyzd[:, 3].min().item())

        self.octree.allocate_points(xyzd, debug=False)

        # clear the cache
        self.cached_xyzd = []

        return min_octree_depth
        