import torch
from typing import *
import utils3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
def normalize_extrinsics(extrinsics : torch.Tensor, scale : Union[float, torch.Tensor], mu : torch.Tensor = None) -> torch.Tensor:
    if mu is None:
        mu = torch.zeros(3, dtype=extrinsics.dtype, device=extrinsics.device)
    extrinsics = extrinsics.clone() # do not change the origin extrinsics !!!
    if len(extrinsics.shape) == 2:
        assert 4 == extrinsics.shape[0] == extrinsics.shape[1]
        R = extrinsics[:3, :3]
        T = extrinsics[:3, 3]
        normed_T = (T + mu @ R.T) / scale
        extrinsics[:3, 3] = normed_T
        return extrinsics
    else:
        assert 4 == extrinsics.shape[1] == extrinsics.shape[2]
        R = extrinsics[:, :3, :3]
        T = extrinsics[:, :3, 3]
        normed_T = (T + mu @ R.transpose(1, 2)) / scale
        extrinsics[:, :3, 3] = normed_T
        return extrinsics

def normalize_intrinsics(intrinsics : torch.Tensor, H : int, W : int) -> torch.Tensor:
    intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / W
    intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / H
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / W
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / H
    return intrinsics

def transpose_to_first_frames(extrinsics : torch.Tensor) -> torch.Tensor:
    inv_first_extrinsics = torch.inverse(extrinsics[0].clone())
    extrinsics = extrinsics @ inv_first_extrinsics
    return extrinsics

def apply_transforms(xyz : torch.Tensor, extrinsics : torch.Tensor) -> torch.Tensor:
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    return xyz @ R.transpose(-1, -2) + T.unsqueeze(-2)


def get_plucker_ray(extrinsics: torch.Tensor, intrinsics: torch.Tensor, 
        H: int = None, W: int = None,
        uv: torch.Tensor = None,
        raw: bool = False) -> torch.Tensor:
    """
    Args:
        extrinsics: (B, V, 4, 4) extrinsic matrices.
        intrinsics: (B, V, 3, 3) intrinsic matrices.
        H (optional): height of the image.
        W (optional): width of the image.
        uv (optional): (..., n_rays, 2) uv coordinates. Batch shape can be omitted and will be broadcasted to extrinsics and intrinsics.
        
    Returns:
        plucker: (B, V, H, W, 6) plucker coordinates.
    """
    if uv is None:
        assert H is not None and W is not None
        uv = utils3d.torch.uv_map(H, W).flatten(0, 1).to(extrinsics)
        
        
    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)
    with torch.amp.autocast('cuda', enabled=False):
        inv_transformation = (intrinsics @ extrinsics[:, :, :3, :3]).inverse()
        inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)       
    # normalize rays_d
    rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-6)                                         
    rays_o = inv_extrinsics[..., :3, 3] # B, V, 3                       
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW                
    if raw:
        return rays_o.reshape(*extrinsics.shape[:2], H, W, 3), rays_d.reshape(*extrinsics.shape[:2], H, W, 3)                                          
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(*extrinsics.shape[:2], H, W, 6)

def visualize_camera_orbit(extrinsics, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    inv_extrinsics = extrinsics.inverse()
    rays_o = inv_extrinsics[..., :3, 3]      
    x = rays_o[:, 0].cpu().numpy()
    y = rays_o[:, 1].cpu().numpy()
    z = rays_o[:, 2].cpu().numpy()
    ax.scatter(x, y, z, c='r', marker='o', s=5)
    ax.plot(x, y, z, c='b', linewidth=2)  # 连接点的线，蓝色，宽度为 2
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')      
    plt.savefig(save_path)



def crop_images(image : torch.Tensor, 
    crop_size : Union[int, Tuple[int, int]],
    mode='random',
    intrinsics : Union[None, torch.Tensor] = None,
    depth = None,
    crop_left = None,
    crop_top = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Crop images to a fixed size.

    Args:
        images: The [N x 3 x H x W] tensor of images.
        crop_size: The size to crop the images to.
        mode: The crop mode, either 'center' or 'random'.
        (Optional) intrinsics: The [N x 3 x 3] tensor of intrinsics matrices.
        (Optional) depth: The [N x 1 x H x W] tensor of depth maps.

    Returns:
        The [N x 3 x crop_size x crop_size] tensor of cropped images and the [N x 3 x 3] tensor of cropped intrinsics matrices.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    original_h, original_w = image.shape[-2:]
    h, w = crop_size
    crop_h, crop_w = min(int(h / w * original_w), original_h), min(int(w / h * original_h), original_w)
    if crop_left is None or crop_top is None:
        if mode == 'center':
            crop_top, crop_left = original_h // 2 - crop_h // 2, original_w // 2 - crop_w // 2
        if mode == 'random':
            crop_top, crop_left = np.random.randint(0, original_h - crop_h + 1), np.random.randint(0, original_w - crop_w + 1)
    image = F.crop(image, crop_top, crop_left, crop_h, crop_w)
    image = F.resize(image, crop_size, antialias=True)
    result = [image]
    if intrinsics is not None:
        transform = torch.Tensor([
            original_w / crop_w, 0, -crop_left / crop_w,
            0, original_h / crop_h, -crop_top / crop_h,
            0, 0, 1
        ]).reshape(3, 3)
        intrinsics = transform @ intrinsics
        result.append(intrinsics)
    if depth is not None:
        depth = F.crop(depth, crop_top, crop_left, crop_h, crop_w)
        depth = F.resize(depth, crop_size, interpolation=InterpolationMode.NEAREST)
        result.append(depth)
    return tuple(result) if len(result) > 1 else result[0]

def recover_crop_from_intrinsics(image : torch.Tensor, 
    crop_size : Union[int, Tuple[int, int]],
    intrinsics_origin, 
    intrinsics_crop):
    """
    Recover the original intrinsics from the cropped intrinsics.
    Args:
        intrinsics_origin: The [3 x 3] tensor of original intrinsics matrices.
        intrinsics_crop: The [3 x 3] tensor of cropped intrinsics matrices.
    Returns:
        The [3 x 3] tensor of recovered original intrinsics matrices.
    """
    assert intrinsics_origin.shape == (3, 3) and intrinsics_crop.shape == (3, 3), f"Intrinsics must be 3x3 matrices. Got {intrinsics_origin.shape} and {intrinsics_crop.shape}."
    transform = intrinsics_crop @ torch.linalg.inv(intrinsics_origin)
    
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    original_h, original_w = image.shape[-2:]
    h, w = crop_size
    crop_h, crop_w = min(int(h / w * original_w), original_h), min(int(w / h * original_h), original_w)
    crop_left = int(-crop_w * transform[0, 2] + 0.5)
    crop_top = int(-crop_h * transform[1, 2] + 0.5)
    print(f"Recovering crop from intrinsics: crop_left={crop_left}, crop_top={crop_top}, crop_h={crop_h}, crop_w={crop_w}")
    return crop_images(image, crop_size, crop_left=crop_left, crop_top=crop_top)
    
def generate_rotate_world_to_camera_trajectory(num_frames=120, cam_position=np.array([0, 0, 4]), up_vector=np.array([0, 1, 0])):
    poses = []

    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames  # 从 0 到 2π

        # 构造视角朝向
        direction = np.array([np.sin(theta), 0, -np.cos(theta)])  # 视角朝向
        direction = direction / np.linalg.norm(direction)

        # 构造旋转矩阵
        right = np.cross(direction, up_vector)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)

        # 相机坐标系为 (right, up, -direction)
        R_c2w = np.stack([right, up, -direction], axis=1)  # camera-to-world rotation

        # 求 world-to-camera 外参矩阵
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ cam_position

        pose = np.eye(4)
        pose[:3, :3] = R_w2c
        pose[:3, 3] = t_w2c

        poses.append(pose)

    return np.stack(poses)


def generate_right_world_to_camera_trajectory(num_frames: int, step_size: float = 0.01) -> torch.Tensor:
    """
    生成一组相机外参矩阵（world to camera），模拟相机沿着 +X 匀速移动。
    
    Args:
        num_frames (int): 相机帧数
        step_size (float): 每帧相机沿 X 方向移动的距离
        
    Returns:
        torch.Tensor: [num_frames, 4, 4] 相机外参矩阵（world to camera）
    """
    extrinsics = []

    # 固定相机朝向：看向 +Z，up 是 +Y
    R = np.eye(3)  # identity rotation（相机坐标轴与世界坐标轴对齐）

    for i in range(num_frames):
        x = i * step_size
        t = np.array([x, 0.0, 0.0])  # 相机位置

        Rt = np.eye(4)
        Rt[:3, :3] = R.T  # world to camera rotation
        Rt[:3, 3] = -R.T @ t  # world to camera translation

        extrinsics.append(Rt)

    return np.stack(extrinsics, axis=0)


def generate_left_world_to_camera_trajectory(num_frames: int, step_size: float = 0.01) -> torch.Tensor:
    """
    生成一组相机外参矩阵（world to camera），模拟相机沿着 +X 匀速移动。
    
    Args:
        num_frames (int): 相机帧数
        step_size (float): 每帧相机沿 X 方向移动的距离
        
    Returns:
        torch.Tensor: [num_frames, 4, 4] 相机外参矩阵（world to camera）
    """
    extrinsics = []

    # 固定相机朝向：看向 +Z，up 是 +Y
    R = np.eye(3)  # identity rotation（相机坐标轴与世界坐标轴对齐）

    for i in range(num_frames):
        x = -i * step_size
        t = np.array([x, 0.0, 0.0])  # 相机位置

        Rt = np.eye(4)
        Rt[:3, :3] = R.T  # world to camera rotation
        Rt[:3, 3] = -R.T @ t  # world to camera translation

        extrinsics.append(Rt)

    return np.stack(extrinsics, axis=0)


def generate_back_world_to_camera_trajectory(num_frames: int, step_size: float = 0.01) -> torch.Tensor:
    """
    生成一组相机外参矩阵（world to camera），模拟相机沿着 +X 匀速移动。
    
    Args:
        num_frames (int): 相机帧数
        step_size (float): 每帧相机沿 X 方向移动的距离
        
    Returns:
        torch.Tensor: [num_frames, 4, 4] 相机外参矩阵（world to camera）
    """
    extrinsics = []

    # 固定相机朝向：看向 +Z，up 是 +Y
    R = np.eye(3)  # identity rotation（相机坐标轴与世界坐标轴对齐）

    for i in range(num_frames):
        x = -i * step_size
        t = np.array([0.0, 0.0, x])  # 相机位置

        Rt = np.eye(4)
        Rt[:3, :3] = R.T  # world to camera rotation
        Rt[:3, 3] = -R.T @ t  # world to camera translation

        extrinsics.append(Rt)

    return np.stack(extrinsics, axis=0)

def generate_up_world_to_camera_trajectory(num_frames: int, step_size: float = 0.01) -> torch.Tensor:
    """
    生成一组相机外参矩阵（world to camera），模拟相机沿着 +Y 匀速移动。
    
    Args:
        num_frames (int): 相机帧数
        step_size (float): 每帧相机沿 Y 方向移动的距离
        
    Returns:
        torch.Tensor: [num_frames, 4, 4] 相机外参矩阵（world to camera）
    """
    extrinsics = []

    # 固定相机朝向：看向 +Z，up 是 +Y
    R = np.eye(3)  # identity rotation（相机坐标轴与世界坐标轴对齐）

    for i in range(num_frames):
        y = i * step_size
        t = np.array([0.0, -y, 0.0])  # 相机位置

        Rt = np.eye(4)
        Rt[:3, :3] = R.T  # world to camera rotation
        Rt[:3, 3] = -R.T @ t  # world to camera translation

        extrinsics.append(Rt)

    return np.stack(extrinsics, axis=0)

def generate_camera_extrinsics(camera_mode, num_frames=121):
    if camera_mode == "rotation": 
        extrinsics_f33 = torch.Tensor(generate_rotate_world_to_camera_trajectory(num_frames=num_frames))
    elif camera_mode == "right":
        extrinsics_f33 = torch.Tensor(generate_right_world_to_camera_trajectory(num_frames=num_frames))
    elif camera_mode == "left":
        extrinsics_f33 = torch.Tensor(generate_left_world_to_camera_trajectory(num_frames=num_frames))
    elif camera_mode == "backward":
        extrinsics_f33 = torch.Tensor(generate_back_world_to_camera_trajectory(num_frames=num_frames))
    elif camera_mode == "up":
        extrinsics_f33 = torch.Tensor(generate_up_world_to_camera_trajectory(num_frames=num_frames))
    else:
        raise NotImplementedError(f"Camera mode {camera_mode} is not implemented.")
    return extrinsics_f33

def get_cameras_accuracy_l2(
    pred_c2w: torch.Tensor,
    gt_c2w: torch.Tensor,
    to_first_frame: bool = False,
    eps=1e-8,
):
    """
    Camera pose accuracy with L2 (least-squares) scale-only alignment.

    Args:
        pred_Rs: (N, 3, 3) predicted rotation matrices
        gt_Rs:   (N, 3, 3) ground-truth rotation matrices
        pred_ts: (N, 3)    predicted translations
        gt_ts:   (N, 3)    ground-truth translations
        eps:     small value to avoid division by zero

    Returns:
        r_score: mean rotation error
        t_score: mean translation error (after scale alignment)
    """
    def transform_to_first_frame(c2w: torch.Tensor) -> torch.Tensor:
        first_c2w_inv = c2w[0].inverse()
        return torch.matmul(first_c2w_inv.unsqueeze(0), c2w)
    
    if to_first_frame:
        pred_c2w = transform_to_first_frame(pred_c2w)
        gt_c2w = transform_to_first_frame(gt_c2w)
        
    pred_Rs = pred_c2w[:, :3, :3]
    gt_Rs = gt_c2w[:, :3, :3]
    pred_ts = pred_c2w[:, :3, 3]
    gt_ts = gt_c2w[:, :3, 3]
    def compare_rotations(R1, R2):
        cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
        cos_err[cos_err > 1] = 1
        cos_err[cos_err < -1] = -1
        return cos_err.acos() * 180 / np.pi
    R_fixed = pred_Rs
    R_error = compare_rotations(R_fixed, gt_Rs)  # (N,)
    r_score = R_error.mean().item()

    # -------------------------
    # Translation: L2 scale-only alignment
    # -------------------------
    # Solve: min_c sum ||gt_t - c * pred_t||^2
    # c* = sum(gt_t · pred_t) / sum(||pred_t||^2)

    # ensure float64 for numerical stability
    pred_ts_d = pred_ts.double()
    gt_ts_d = gt_ts.double()

    numerator = torch.sum(pred_ts_d * gt_ts_d)
    denominator = torch.sum(pred_ts_d * pred_ts_d) + eps
    c_opt = numerator / denominator

    t_fixed = c_opt * pred_ts_d

    # translation error: mean Euclidean distance
    t_error = torch.linalg.norm(t_fixed - gt_ts_d, dim=-1)
    t_score = t_error.mean().item()

    print(f"Rotation error: {r_score}")
    print(f"Translation error: {t_score}")

    return r_score, t_score
