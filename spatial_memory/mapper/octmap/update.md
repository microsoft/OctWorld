之前的update方法只有两种模式：1. 根据allocate indices 2. update all，现在我需要你写一个新模式，基于camera view frustrum。做法：在dynoctree类里面新写一个函数，获取activate nodes based on view furstrum
做法：
从full depth 层出发，判定本层的cubes是否在view frstrum内，如果在，
1. 叶子节点：加入update list
2. 非叶子节点：加入allocate list，继续往下层判定
最后返回所有需要更新的叶子节点。

判定方法Hint:
把camera ext和intrin 变成4个平面，做AABB boundingbox 判定，示例如下：

```python
import torch


def frustum_intersection(
    centers,      # (B,3)
    extents,      # (B,3)
    K,            # (3,3)
    W,
    H,
    R,
    t,
):
    """
    robust frustum-AABB test
    """

    # world -> camera
    centers_cam = centers @ R.T + t

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    device = centers.device

    # 4 planes
    normals = torch.tensor([
        [ fx, 0,  cx],         # left
        [-fx, 0, W-cx],        # right
        [0,  fy, cy],          # top
        [0, -fy, H-cy],        # bottom
    ], device=device).float()

    # normalize
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # (B,4)
    s = centers_cam @ normals.T

    # projected radius
    # (B,4,3)
    abs_n = normals.abs()[None,:,:]

    r = (extents[:,None,:] * abs_n).sum(dim=-1)

    outside = (s + r) < 0

    reject = outside.any(dim=1)

    return ~reject
```

Hint： 
1. 你需要仔细第考虑这个函数里面extrinsics和intrinsics的输入格式，以及坐标系约定，确保他和我们mapper里面使用方法一致。
2. 由于我的相机有很多个view，索引操作比较慢。一个可用的建议是：设定一个camera view number的batch size，每次同时把多个视角的frustum plane算出来，然后对每个cube做一次intersection test，如果这个cube在任意一个view的frustum内，就加入update list或者allocate list。 这个设定成一个可变的参数，默认是20。算法流程类似。
allocate(原来的function).....; 
for i, batch in enumerate(camera_batches):
    fetch active nodes based on batch of camera views
    for each view in batch:
        update this view's tsdf values for active nodes


为了支持这个功能，你还需要把上面的这个python 函数改成支持多plane 并行的版本（本质还是矩阵乘法+布尔判断）

最后，把原有的allocate_indices的这个模式去掉，这个模式不太行，只保留update_all和view_frustum两种模式。

你还需要完成一个test time 测试，具体方法就是先integrate一些帧；构建完octree之后把一个新的相机 frsturm 切出来的octree visualize 出来，并且也保存原有的integration mesh。在example/下完成这个事情