import torch
from typing import *
import utils3d

class PinholeCamera:
    def __init__(
        self, 
        extrinsics: torch.Tensor, 
        intrinsics: torch.Tensor, 
    ):
        """
        A Single View PinholeCamera Parametrization
        Args:
            extrinsics: (4, 4) World to Camera
            intrinsics: (3, 3) Camera Intrinsics
        """
        self.extrinsics = extrinsics  # (4, 4)
        self.intrinsics = intrinsics  # (3, 3)
        
    @property
    def camera_origin(self):
        '''
        Get camera origin in world coordinates
        Returns:
            (3,) Camera origin
        '''
        inv_extrinsics = torch.inverse(self.extrinsics)
        camera_origin = inv_extrinsics[:3, 3]
        return camera_origin
        
    def back_project_points(self, depths, return_pixel_sizes: bool = False):
        '''
        Backproject depth map to point cloud
        Args:
            depths: (H, W) Depth map
            return_pixel_sizes: whether to return pixel sizes
        Returns:
            pts: (H, W, 3) Point cloud
            pixel_sizes: (H, W) Pixel sizes in the 3d global space if return_pixel_sizes is True
        '''
        height, width = depths.shape[-2:]
        pts = utils3d.depth_map_to_point_map(depths, self.intrinsics, self.extrinsics)
        if not return_pixel_sizes:
            return pts
        
        fx = self.intrinsics[0, 0] * width
        fy = self.intrinsics[1, 1] * height
        pixel_sizes = depths / torch.sqrt(fx * fy)
        return pts, pixel_sizes
    
    def project_points(self, points : torch.Tensor):
        '''
        Project 3D points to 2D pixel coordinates
        Args:
            points: (N, 3) 3D points in world coordinates
        Returns:
            pixel_uv: (N, 2) Pixel coordinates in normalized [0, 1] range
            linear_depths: (N,) Linear depths of the points
            valid_mask: (N,) Boolean mask indicating whether the points are valid (in front of camera and within image bounds)
        '''
        pixel_uv, linear_depths = utils3d.project_cv(points, self.intrinsics, self.extrinsics)
        valid_mask = (pixel_uv[:, 0] >= 0) & (pixel_uv[:, 0] < 1) & \
                     (pixel_uv[:, 1] >= 0) & (pixel_uv[:, 1] < 1) & \
                     (linear_depths > 0)
        return pixel_uv, linear_depths, valid_mask
    
    def calc_volume_pixel_size(self, linear_depths: torch.Tensor, volume_size: float):
        '''
        Calculate the pixel sizes in the 3D global space for a given volume size at certain linear depths
        Args:
            linear_depths: (N,) Linear depths of the points
            volume_size: The size of the volume voxel
        Returns:
            pixel_sizes: (N,) Pixel sizes in the normalized 2D pixel space [0-1] range
        '''
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        # h_size = fx * volume_size / linear_depths, w_size = fy * volume_size / linear_depths
        pixel_sizes = torch.sqrt(fx * fy) * volume_size / linear_depths
        return pixel_sizes
        
        