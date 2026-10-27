import torch

def split_cube_features(features):
    """
    将 (N, 8, C) 的特征插值为 (N, 8, 8, C)，表示立方体分裂后的 8 个子立方体特征。
    
    Args:
        features (torch.Tensor): 输入特征，形状为 (N, 8, C)
                                索引 0-7 遵循 Morton 编码 (x, y, z):
                                0: 000 (0,0,0)
                                1: 001 (1,0,0)
                                2: 010 (0,1,0)
                                3: 011 (1,1,0)
                                ...
    Returns:
        torch.Tensor: 输出特征，形状为 (N, 8, 8, C)
                      dim 1: 子立方体索引 (0-7, Morton)
                      dim 2: 每个子立方体的角点索引 (0-7, Morton)
    """
    N, _, C = features.shape
    device = features.device
    dtype = features.dtype

    # 1. 定义原立方体 8 个角点的坐标 (Morton 顺序: x是高位, y是中位, z是低位)
    corner_indices = torch.arange(8, device=device)
    morton_coords = torch.stack([
        (corner_indices >> 2) & 1, # x
        (corner_indices >> 1) & 1, # y
        (corner_indices >> 0) & 1  # z
    ], dim=-1).to(dtype)
    
    child_coords = (morton_coords.unsqueeze(0) + morton_coords.unsqueeze(1)) / 2.0  # (8, 8, 3)
    child_coords = child_coords.reshape(64, 3)  # (64, 3)

    # 3. 计算三线性插值权重矩阵 (64, 8)
    diff = torch.abs(child_coords.unsqueeze(1) - morton_coords.unsqueeze(0)) # (64, 8, 3)
    weights = torch.prod(1.0 - diff, dim=-1) # (64, 8)
    # weights = weights.unsqueeze(0).expand(N, -1, -1)  # (N, 64, 8)
    # if split_mask is not None:
    #     # split mask is like (N, 8), where 1 means this corner is valid and should contribute to the children, 0 means this corner is invalid and should not contribute to the children
    #     weights = weights * split_mask.unsqueeze(1)
    #     # normalize weights so that for each child, the sum of weights from valid corners is 1
    #     weights_sum = weights.sum(dim=-1, keepdim=True)  # (N, 64, 1)
    #     standard_weights = torch.ones_like(weights) / 8.0
    #     weights = torch.where(weights_sum > 0, weights / weights_sum, standard_weights)  # avoid division by zero

    # 4. 执行插值
    res = torch.einsum('ij,njc->nic', weights, features)
    
    return res.reshape(N, 8, 8, C)


def split_cube_features_nearest(features):
    """
    将 (N, 8, C) 的特征用 nearest 插值为 (N, 8, 8, C)

    Args:
        features (torch.Tensor): (N, 8, C)

    Returns:
        torch.Tensor: (N, 8, 8, C)
    """
    N, _, C = features.shape
    device = features.device
    dtype = features.dtype

    # 原 cube 8 个 corner 的 Morton 坐标
    corner_indices = torch.arange(8, device=device)
    morton_coords = torch.stack([
        (corner_indices >> 2) & 1,  # x
        (corner_indices >> 1) & 1,  # y
        (corner_indices >> 0) & 1   # z
    ], dim=-1).to(dtype)  # (8, 3)

    # 子 cube 的 8x8 个角点坐标
    child_coords = (morton_coords.unsqueeze(0) + morton_coords.unsqueeze(1)) / 2.0
    child_coords = child_coords.reshape(64, 3)  # (64, 3)

    # 🔹 Nearest: 四舍五入到最近 corner
    nearest_coords = torch.round(child_coords)  # (64, 3)

    # 🔹 把 (x,y,z) 转回 Morton index
    nearest_indices = (
        (nearest_coords[:, 0].long() << 2) |
        (nearest_coords[:, 1].long() << 1) |
        (nearest_coords[:, 2].long())
    )  # (64,)

    # 🔹 直接 gather 对应 corner 特征
    res = features[:, nearest_indices, :]  # (N, 64, C)

    return res.reshape(N, 8, 8, C)


# --- 测试代码 ---
if __name__ == "__main__":
    # 假设输入特征是角点坐标本身，便于验证插值结果
    # 000 -> [0,0,0], 111 -> [1,1,1]
    input_feats = torch.tensor([
        (
        (v >> 2) & 1, # x
        (v >> 1) & 1, # y
        (v >> 0) & 1  # z
        ) for v in range(8)], dtype=torch.float32).unsqueeze(0)  # (1, 8, 3)

    output = split_cube_features(input_feats)  # (1*8, 8, 3)
    device="cpu"
    parent_xyz = input_feats[:1, 0].long()
    morton_directions = torch.arange(8, device=device)
    xyz_directions = torch.stack([
        (morton_directions >> 2) & 1,
        (morton_directions >> 1) & 1,
        morton_directions & 1
    ], dim=-1)  # (8, 3)
    print(parent_xyz)
    new_child_xyz = (parent_xyz << 1).unsqueeze(1) + xyz_directions.unsqueeze(0)  # (N, 8, 3)
    new_child_xyz = new_child_xyz.reshape(-1, 3)
    cube_offsets = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
    ], device=device, dtype=torch.long)  # (8, 3)
    points = new_child_xyz.unsqueeze(1) + cube_offsets.unsqueeze(0)  # (N, 8, 3)
    points = points / 2.0
    print(torch.allclose(output, points))