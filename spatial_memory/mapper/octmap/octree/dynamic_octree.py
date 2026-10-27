import torch
from typing import *
from .shuffled_key import key2xyz, xyz2key
from ..frustum import build_frustum_planes, frustum_aabb_intersection
import time
from ..utils import split_cube_features, split_cube_features_nearest


def construct_full_octree(
    full_depth: int,
    feature_dims: int,
    device: Union[str, torch.device] = "cuda",
):
    """
    Construct a full octree from depth 0 to full_depth.

    Returns:
        xyz: List[Tensor], each (N_d, 3)
        children: List[Tensor], each (N_d,)
        parent: List[Optional[Tensor]], each (N_d,) except depth 0 is None
        features: List[Tensor], each (N_d, 8, C)
        weights: List[Tensor], each (N_d, 8)
    """
    xyz = [None] * (full_depth + 1)
    children = [None] * (full_depth + 1)
    parent = [None] * (full_depth + 1)
    features = [None] * (full_depth + 1)
    weights = [None] * (full_depth + 1)

    for depth in range(full_depth + 1):
        num = 1 << (3 * depth)
        morton_keys = torch.arange(num, dtype=torch.long, device=device)

        if depth != full_depth:
            children[depth] = torch.arange(num, dtype=torch.long, device=device)
        else:
            children[depth] = torch.full((num,), -1, dtype=torch.long, device=device)

        features[depth] = torch.ones((num, 8, feature_dims), device=device)
        weights[depth] = torch.zeros((num, 8), device=device)

        if depth != 0:
            parent[depth] = torch.arange(num >> 3, dtype=torch.long, device=device)

        x, y, z, _ = key2xyz(morton_keys, depth)
        xyz[depth] = torch.stack([x, y, z], dim=-1)

    return xyz, children, parent, features, weights

class DynOctree:
    def __init__(self, 
            feature_dims=4, 
            depth=16, 
            full_depth=2, 
            interplation_mode="nearest",
            device="cuda"):
        self.feature_dims = feature_dims
        self.depth = depth
        self.full_depth = full_depth
        self.device = device
        self.interplation_mode = interplation_mode
        
        # properties after building the octree
        self.reset()
        
        # grow the octree structure (to be done)
        self.init_octree(full_depth)
        
    def init_octree(self, full_depth):
        xyz, children, parent, features, weights = construct_full_octree(
            full_depth=full_depth,
            feature_dims=self.feature_dims,
            device=self.device,
        )

        for depth in range(full_depth + 1):
            self.xyz[depth] = xyz[depth]
            self.children[depth] = children[depth]
            # self.parent[depth] = parent[depth]
            self.features[depth] = features[depth]
            self.weights[depth] = weights[depth]
            self.nnum[depth] = 1 << (3 * depth)
        self.cur_depth = full_depth
        
    def extend_root(self):
        '''
        Extend the octree structure by adding a new root node and making the old root node as its child.
        the overall scene will scaled by 0.5x

        Minimal seam-only strategy:
        - Build a new full octree with depth=2 as scaffold.
        - Embed old depth-1 nodes into new depth-2 nodes (position correspondence).
        - Shift old depth>=2 tensors to depth+1 directly, without modifying their
          children/features/weights topology.
        '''

        old_cur_depth = self.cur_depth
        old_xyz = self.xyz
        old_children = self.children
        # old_parent = self.parent
        old_features = self.features
        old_weights = self.weights
        old_nnum = self.nnum.clone()

        num_levels = self.depth + 1 + 1
        self.xyz = [None] * num_levels
        self.children = [None] * num_levels
        # self.parent = [None] * num_levels
        self.features = [None] * num_levels
        self.weights = [None] * num_levels
        self.nnum = torch.zeros(num_levels, dtype=torch.long)

        # 1) New scaffold at top 3 levels.
        base_xyz, base_children, base_parent, base_features, base_weights = construct_full_octree(
            full_depth=2,
            feature_dims=self.feature_dims,
            device=self.device,
        )
        for d in range(3):
            self.xyz[d] = base_xyz[d]
            self.children[d] = base_children[d]
            # self.parent[d] = base_parent[d]
            self.features[d] = base_features[d]
            self.weights[d] = base_weights[d]
            self.nnum[d] = self.xyz[d].shape[0]

        # 2) Place old depth-1 into corresponding new depth-2 nodes.
        # We place old tree into child-0 subspace of new root, so xyz correspondence is identity at this seam.
        # xyz mapping relation new_xyz = old_xyz + 2^(old_depth - 1)
        seam_indices = xyz2key(
            old_xyz[1][:, 0] + 1,
            old_xyz[1][:, 1] + 1,
            old_xyz[1][:, 2] + 1,
            2,
        ).long()
        # NOTE: do NOT overwrite self.xyz[2][seam_indices]; the scaffold already
        # holds the correct new-coords (= old_xyz[1] + 1). Writing old_xyz[1]
        # here would corrupt xyz at those positions.
        self.features[2][seam_indices] = old_features[1]
        self.weights[2][seam_indices] = old_weights[1]
        self.children[2][seam_indices] = old_children[1]

        # 3) Shift old depth>=2 to new depth>=3 directly.
        for old_d in range(2, old_cur_depth + 1):
            new_d = old_d + 1
            self.xyz[new_d] = old_xyz[old_d] + (2 ** (old_d - 1))
            self.children[new_d] = old_children[old_d]
            self.features[new_d] = old_features[old_d]
            self.weights[new_d] = old_weights[old_d]
            # self.parent[new_d] = old_parent[old_d]
            self.nnum[new_d] = old_nnum[old_d]
            
        # features mapping relation sdf /= 2.0
        for d in range(2, old_cur_depth + 1):
            self.features[d][:, :, 0] = self.features[d][:, :, 0] / 2.0

        # # 4) Only fix seam parent relation: old depth-2 parents (0..7) -> seam indices in new depth-2.
        # if old_cur_depth >= 2 and self.parent[3] is not None:
        #     seam_map = seam_indices
        #     self.parent[3] = seam_map[self.parent[3].long()]

        self.cur_depth = old_cur_depth + 1
        self.depth = self.depth + 1

    def reset(self):
        ''' 
            Resets the Octree status and constructs several lookup tables.
        '''
        # octree features in each octree layers
        num = self.depth + 1
        self.root_index = 0                 # the root index of the octree
        self.xyz = [None] * num            # (N, 3) the left bottom front cube position and depth of octree nodes
        self.children = [None] * num        # (N,) children indices of octree nodes, the parent node u will have 8 children nodes, the child[u] << 8 + i is the i-th child node of u
        # self.parent = [None] * num          # parent indices of octree nodes, shape (N_d,)
        self.features = [None] * num        # features of octree nodes, shape (N_d, 8, C) the feature at 8 corners
        self.weights = [None] * num         # weights of octree nodes, shape (N_d, 8) the weights at 8 corners

        # octree node numbers in each octree layers.
        # These are small 1-D tensors; just keep them on CPUs
        self.nnum = torch.zeros(num, dtype=torch.long)

    def reset_observations(self):
        '''
        Reset all TSDF observations while keeping the octree structure intact.

        Zeros out every existing layer's observation ``weights`` and restores
        ``features`` to their initial value (``1.0``, matching
        ``construct_full_octree``). The spatial topology (``xyz`` / ``children``
        / ``nnum`` / ``cur_depth``) is left untouched.

        This is used by the offline integration mode: allocate the octree over
        all frames, wipe accumulated observations, then re-update all frames from
        a clean slate without rebuilding the tree.
        '''
        for depth in range(self.cur_depth + 1):
            if self.weights[depth] is not None:
                self.weights[depth].zero_()
            if self.features[depth] is not None:
                self.features[depth].fill_(1.0)

    def pixel_size_to_volume_size(self, pixel_sizes: torch.Tensor):
        '''
        Convert pixel sizes to volume sizes and octree depths
        Args:
            pixel_sizes: (N,) Pixel sizes in the 3d global space
        Returns:
            volume_sizes: (N,) Volume sizes in the octree
            octree_depths: (N,) Octree depths for each pixel size
        '''
        # the voxel size at depth d is 2^(1-d)
        octree_depths = torch.clamp((-(pixel_sizes + 1e-6).log2()).floor().long(), min=0, max=self.depth)
        volume_sizes = 2.0 ** (1-octree_depths)
        return volume_sizes, octree_depths
    
    def get_voxel_size(self, depth: int):
        '''
        Get the voxel size at the specified depth
        Args:
            depth: The octree depth
        Returns:
            voxel_size: The voxel size at the specified depth
        '''
        return 2.0 ** (1-depth)
    
    @staticmethod
    def save_cat(x, y):
        if x is None:
            return y
        else:
            return torch.cat([x, y], dim=0)
        
    def debug_split_to_next_lod(self):
        '''
        allocate the last layer nodes all children
        '''
        for d in range(self.cur_depth + 1, 1, -1):
            if self.children[d] is None:
                continue
            active_mask = (self.children[d] == -1)
            parent_indices = torch.nonzero(active_mask, as_tuple=True)[0]
            self.allocate_children(parent_indices, d, split_weight_decay=1.0)
            
    
    def allocate_children(self, parent_indices : torch.Tensor, depth: int, split_weight_decay=1.0):
        '''
        Allocate children nodes for the given parent nodes at the specified depth
        Args:
            parent_indices: (N,) The parent indices
            depth: The current octree depth
        Returns:
            new_node_indices: (M,) Indices of all newly allocated child nodes at depth+1,
                or empty tensor if nothing was allocated.
        '''
        device = self.device
        parent_indices = torch.unique(parent_indices, sorted=False)
        parent_indices = parent_indices[self.children[depth][parent_indices] == -1] # only allocate for parents without children
        if parent_indices.shape[0] == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)
        # print("Allocating", parent_indices.shape[0], "parents at depth", depth)
        allocate_node_num = parent_indices.shape[0] * 8
        current_num_nodes = self.nnum[depth + 1].item()
        new_child_indices = torch.arange(current_num_nodes >> 3, (current_num_nodes + allocate_node_num) >> 3, device=device)
        
        parent_features = self.features[depth][parent_indices]  # (N, 8, C)
        parent_weights = self.weights[depth][parent_indices]  # (N, 8)
        interploation_func = split_cube_features_nearest if self.interplation_mode == "nearest" else None #split_cube_features
        # print(split_weight_decay, self.interplation_mode)
        new_features = interploation_func(parent_features) # (N, 8, 8, C)
        new_weights = interploation_func(parent_weights.unsqueeze(-1)).squeeze(-1) * split_weight_decay # (N, 8, 8)
        # new_weights = torch.clamp(new_weights, max=1.0)  # avoid overflow
        
        # children_sdf_truncate_margin = self.get_voxel_size(depth + 1) * self.sdf_trunc_ratio
        # new_features[..., 0] = torch.clamp(new_features[..., 0], max=children_sdf_truncate_margin, min=-children_sdf_truncate_margin)  # clamp the sdf values to avoid large values that may cause instability
        new_features = new_features.flatten(0, 1)  # (N*8, 8, C)
        new_weights = new_weights.flatten(0, 1)  # (N*8, 8)
        
        new_children = torch.full((allocate_node_num, ), -1, dtype=torch.long, device=device)
        
        parent_xyz = self.xyz[depth][parent_indices]  # (N, 3)
        morton_directions = torch.arange(8, device=device)
        xyz_directions = torch.stack([
            (morton_directions >> 2) & 1,
            (morton_directions >> 1) & 1,
            morton_directions & 1
        ], dim=-1)  # (8, 3)
        new_child_xyz = (parent_xyz << 1).unsqueeze(1) + xyz_directions.unsqueeze(0)  # (N, 8, 3)
        new_child_xyz = new_child_xyz.reshape(-1, 3)
        # new_parent_indices = parent_indices.unsqueeze(1).repeat(1, 8).reshape(-1) # (N*8,)
        
        # extend attributes
        self.features[depth + 1] = self.save_cat(self.features[depth + 1], new_features)
        self.weights[depth + 1] = self.save_cat(self.weights[depth + 1], new_weights)
        self.xyz[depth + 1] = self.save_cat(self.xyz[depth + 1], new_child_xyz)
        self.children[depth + 1] = self.save_cat(self.children[depth + 1], new_children)
        
        # link the children to parents
        self.children[depth][parent_indices] = new_child_indices
        # self.parent[depth + 1] = self.save_cat(self.parent[depth + 1], new_parent_indices)
        self.nnum[depth + 1] += allocate_node_num
        
        self.cur_depth = max(self.cur_depth, depth + 1)
        
        
    def keys_to_points(self, keys: torch.Tensor, depth: int):
        '''
        Convert the given keys to 3D points in global coordinates
        Args:
            keys: (N,) The shuffled keys to be converted
            depth: (int) The octree depth for the keys
        Returns:
            points: (N, 3) The 3D points in global coordinates
        '''
        x, y, z, b = key2xyz(keys, depth)
        points = torch.stack([x, y, z], dim=-1).float()
        points = points / (2 ** (depth - 1)) - 1.0
        center_offset = 0.5 * self.get_voxel_size(depth)
        points = points + center_offset
        return points
    
    def points_to_xyzd(self, points: torch.Tensor, depths: Union[int, torch.Tensor], return_unique=False):
        '''
        Convert the given 3D points in global coordinates to shuffled keys
        Args:
            points: (N, 3) The 3D points in global coordinates
            depths: (int) or (N,) The octree depths for the points
            return_unique: Whether to return unique keys
        Returns:
            xyzd: (N, 4) The grid positions in the voxels with depths
        '''
        assert (points > -1.0).all() and (points < 1.0).all(), "Points are out of bounds [-1, 1]"
        points = (points + 1.0) * 2 ** (depths[:, None] - 1)  # normalize points to [0, 2^depth]
        xyzd = torch.cat([points.long(), depths.unsqueeze(-1)], dim=-1)
        if return_unique:
            xyzd = torch.unique(xyzd, dim=0)
        return xyzd
    
    def xyz_to_points(self, xyz: torch.Tensor, depths: Union[int, torch.Tensor]):
        '''
        Convert the given grid positions in the voxels with depths to 3D points in global coordinates
        Args:
            xyz: (N, 3) The grid positions in the voxels with depths
            depths: (int) or (N,) The octree depths for the points
        Returns:
            points: (N, 8, 3) The 3D points in global coordinates (a cube of 8 points for each voxel)
        '''
        
        cube_offsets = torch.tensor(
            [((v >> 2) & 1, (v >> 1) & 1, v & 1) for v in range(8)], device=xyz.device, dtype=torch.long)
        points = xyz.unsqueeze(1) + cube_offsets.unsqueeze(0)  # (N, 8, 3)
        points = points / (2 ** depths) * 2.0 - 1.0 # normalize points to [-1, 1]
        return points
    
    def construct_neighbor_xyzd(self, point_xyzd : torch.Tensor, extend_size=2):
        '''
        Construct the 26 neighbor keys for the given node keys
        Args:
            point_xyzd: (N, 4) The grid positions of the nodes with depths
            extend_size: The size to extend the neighbors
        Returns:
            point_xyzd: (M,) The neighbor keys with depths
        '''
        cube_size = extend_size * 2 + 1
        cube = torch.ones((cube_size, cube_size, cube_size), device=point_xyzd.device, dtype=torch.long)  # (cube_size, cube_size, cube_size)
        neighbor_grid = torch.nonzero(cube, as_tuple=False) - extend_size  # (cube_size**3, 3)
        
        xyz = point_xyzd[:, :3].unsqueeze(1)  # (N, 1, 3)
        neighbor_xyz = xyz + neighbor_grid.unsqueeze(0)  # (N, cube_size**3, 3)
        d = point_xyzd[:, 3].unsqueeze(1).repeat(1, cube_size**3)
        
        neighbor_xyz = neighbor_xyz.reshape(-1, 3)
        d = d.reshape(-1)
        
        valid_mask = (neighbor_xyz >= 0).all(dim=-1) & (neighbor_xyz < ((1 << d) - 1).unsqueeze(-1)).all(dim=-1)
        neighbor_xyz = neighbor_xyz[valid_mask]
        d = d[valid_mask]
        neighbor_xyzd = torch.cat([neighbor_xyz, d.unsqueeze(-1)], dim=-1)
        neighbor_xyzd = torch.unique(neighbor_xyzd, dim=0)
        return neighbor_xyzd
    
    def allocate_points(self, points_xyzd: torch.Tensor, debug=False):
        '''
        Allocate points into the octree structure
        Args:
            points_xyzd: (N, 4) Points in global coordinates with depths
        '''
        
        # get all the xyzd of the points at the sepcified depths
        # print(" Maximum octree depth for allocation: ", octree_depths.max().item())
        # points_xyzd = self.points_to_xyzd(points, octree_depths)
        
        current_nodes = torch.tensor([self.root_index], device=self.device).repeat(points_xyzd.shape[0])  # start from the root node
        
        # from top to bottom : allocate the octree structure
        # notice that at the last depth, we do not need to allocate children
        for d in range(self.depth):
            # check the position
            if debug:
                debug_points_at_depth = points_xyzd[points_xyzd[:, 3] == d]
                # print("At exactly depth", d, "there are", debug_points_at_depth.shape[0], "points to allocate")
                if debug_points_at_depth.shape[0] > 0:
                    debug_nodes_at_depth = current_nodes[points_xyzd[:, 3] == d]
                    debug_points_xyz_on_tree = self.xyz[d][debug_nodes_at_depth]
                    if not ((debug_points_at_depth[:, :3] == debug_points_xyz_on_tree).all()):
                        print("Error: Point positions do not match at depth", d)
                        # print example points
                        not_align_mask = (debug_points_at_depth[:, :3] != debug_points_xyz_on_tree).any(dim=-1)
                        print("Example mismatched points:")
                        print("Expected:", debug_points_at_depth[not_align_mask][:5])
                        print("Got:", debug_points_xyz_on_tree[not_align_mask][:5])
                        raise ValueError("Point positions do not match the octree structure.")
                
            # filter the points that need to go deeper
            current_nodes = current_nodes[points_xyzd[:, 3] > d]
            points_xyzd = points_xyzd[points_xyzd[:, 3] > d]
            # print("At depth", d, "allocating", points_xyzd.shape[0], "points")
            if points_xyzd.shape[0] == 0:
                break
            self.allocate_children(current_nodes, d)
            assert self.children[d][current_nodes].min() >= 0, "Some parent nodes do not have children after allocation!"
            
            next_layer_directions = (points_xyzd[:, :3] >> (points_xyzd[:, 3].unsqueeze(-1) - 1 - d)) & 1  # (N, 3) means the direction to go to next layer
            next_layer_directions = next_layer_directions[:, 0] << 2 | next_layer_directions[:, 1] << 1 | next_layer_directions[:, 2]  # (N,), the bias to fetch the next layer child
            current_nodes = self.children[d][current_nodes] << 3 | next_layer_directions  # (N,)


    def clean_memory(self, weight_threshold: float = 0.0):
        '''
        Bottom-up parallel memory cleanup.

        If one parent cube's 8 children all satisfy:
          1) each child is leaf (no grandchildren), and
          2) all 8 corner weights of all 8 children <= weight_threshold,
        then remove the whole child octet together.

        This function performs per-depth batched pruning and index remapping.
        '''
        if self.cur_depth <= 0:
            return

        device = self.device
        octant_offsets = torch.arange(8, device=device, dtype=torch.long)

        for d in range(self.cur_depth - 1, -1, -1):
            if self.children[d] is None or self.nnum[d + 1].item() == 0:
                continue

            parent_has_children = self.children[d] >= 0
            if not parent_has_children.any():
                continue

            parent_indices = torch.nonzero(parent_has_children, as_tuple=False).squeeze(-1)
            child_base = self.children[d][parent_indices]  # (P,)
            child_indices = (child_base.unsqueeze(1) << 3) + octant_offsets.unsqueeze(0)  # (P, 8)

            child_weights = self.weights[d + 1][child_indices]  # (P, 8, 8)
            all_child_corners_small = (child_weights <= weight_threshold).all(dim=-1).all(dim=-1)  # (P,)

            child_leaf_mask = (self.children[d + 1][child_indices] == -1).all(dim=-1)  # (P,)
            prune_parent_mask = all_child_corners_small & child_leaf_mask
            if not prune_parent_mask.any():
                continue

            prune_parent_indices = parent_indices[prune_parent_mask]
            prune_child_indices = child_indices[prune_parent_mask].reshape(-1)

            old_num = self.nnum[d + 1].item()
            keep_mask = torch.ones(old_num, dtype=torch.bool, device=device)
            keep_mask[prune_child_indices] = False

            old_to_new = torch.full((old_num,), -1, dtype=torch.long, device=device)
            kept_old_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            old_to_new[kept_old_indices] = torch.arange(kept_old_indices.shape[0], device=device, dtype=torch.long)

            # Compact next depth tensors.
            self.xyz[d + 1] = self.xyz[d + 1][keep_mask]
            self.features[d + 1] = self.features[d + 1][keep_mask]
            self.weights[d + 1] = self.weights[d + 1][keep_mask]
            self.children[d + 1] = self.children[d + 1][keep_mask]
            self.nnum[d + 1] = keep_mask.sum().to(self.nnum.dtype)

            # Remap current depth child-base pointers to compacted indices.
            old_child_base_all = self.children[d].clone()
            valid_base_mask = old_child_base_all >= 0
            if valid_base_mask.any():
                first_child_old_idx = old_child_base_all[valid_base_mask] << 3
                first_child_new_idx = old_to_new[first_child_old_idx]
                self.children[d][valid_base_mask] = first_child_new_idx >> 3

            # Explicitly set pruned parents to leaf.
            self.children[d][prune_parent_indices] = -1

        while self.cur_depth > 0:
            if self.xyz[self.cur_depth] is None or self.nnum[self.cur_depth].item() == 0:
                self.cur_depth -= 1
            else:
                break
            
    def fetch_active_points(self, weight_threshold=0.0):
        """
        Fetch all the leaf nodes (childrenless nodes).

        Returns
            features (N_d, C): The features of the active octree nodes.
            weights (N_d,): The weights of the active octree nodes.
            positions (N_d, 3): The positions of the active octree nodes.
            voxel_sizes (N_d, ): The voxel sizes of the active octree nodes.
        """
        features = []
        weights = []
        active_indices = []
        positions = []
        voxel_sizes = []
        # points = (points + 1.0) * 2 ** (self.depth - 1)  # normalize points to [0, 2^depth]
        for d in range(self.cur_depth + 1):
            active_mask = (self.children[d] == -1) & (self.weights[d] >= weight_threshold).any(dim=-1)
            active_indices.append(torch.nonzero(active_mask, as_tuple=True)[0])
            features.append(self.features[d][active_mask])
            weights.append(self.weights[d][active_mask])
            positions.append(self.xyz_to_points(self.xyz[d][active_mask], d))
            voxel_size = self.get_voxel_size(d)
            voxel_sizes.append(torch.full((len(active_indices[-1]), 8), voxel_size, device=self.device))
        
        self.cached_active_indices = active_indices
        
        
        return torch.cat(features, dim=0).reshape(-1, self.feature_dims), \
               torch.cat(weights, dim=0).reshape(-1), \
               torch.cat(positions, dim=0).reshape(-1, 3), \
                torch.cat(voxel_sizes, dim=0).reshape(-1)
    
    def update(self, features: torch.Tensor, weights: torch.Tensor):
        '''
        Update the features and weights of the given octree nodes at the specified depth
        Args:
            features: (N_d, C) The updated features
            weights: (N_d,) The updated weights
        '''
        for d, idx in enumerate(self.cached_active_indices):
            num_active_nodes = len(idx)
            if num_active_nodes == 0:
                continue
            updated_features = features[:num_active_nodes * 8].reshape(-1, 8, self.feature_dims)
            updated_weights = weights[:num_active_nodes * 8].reshape(-1, 8)
            self.features[d][idx] = updated_features
            self.weights[d][idx] = updated_weights
            features = features[num_active_nodes * 8:]
            weights = weights[num_active_nodes * 8:]
        assert features.shape[0] == 0, "Some features are not updated!"
    
    def fetch_points_by_indices(self, per_depth_indices, leaf_only=True):
        '''
        Fetch features, weights, positions, and voxel_sizes for nodes specified by per-depth indices.
        Also sets cached_active_indices so that update() can write back to these nodes.
        
        Args:
            per_depth_indices: List[Tensor], one per depth, containing node indices
            leaf_only: If True, filter out non-leaf nodes (children != -1)
        Returns:
            features (N_d, C): The features of the specified octree nodes.
            weights (N_d,): The weights of the specified octree nodes.
            positions (N_d, 3): The positions of the specified octree nodes.
            voxel_sizes (N_d,): The voxel sizes of the specified octree nodes.
        '''
        features = []
        weights = []
        positions = []
        voxel_sizes = []
        active_indices = [torch.tensor([], device=self.device, dtype=torch.long) for _ in range(self.cur_depth + 1)]
        
        for d in range(self.cur_depth + 1):
            if d >= len(per_depth_indices):
                break
            idx = per_depth_indices[d]
            if len(idx) == 0:
                continue
            if leaf_only:
                leaf_mask = self.children[d][idx] == -1
                idx = idx[leaf_mask]
                if len(idx) == 0:
                    continue
            active_indices[d] = idx
            features.append(self.features[d][idx])
            weights.append(self.weights[d][idx])
            positions.append(self.xyz_to_points(self.xyz[d][idx], d))
            voxel_size = self.get_voxel_size(d)
            voxel_sizes.append(torch.full((len(idx), 8), voxel_size, device=self.device))
        
        self.cached_active_indices = active_indices
        
        if len(features) == 0:
            return (torch.zeros(0, self.feature_dims, device=self.device),
                    torch.zeros(0, device=self.device),
                    torch.zeros(0, 3, device=self.device),
                    torch.zeros(0, device=self.device))
        
        return torch.cat(features, dim=0).reshape(-1, self.feature_dims), \
               torch.cat(weights, dim=0).reshape(-1), \
               torch.cat(positions, dim=0).reshape(-1, 3), \
               torch.cat(voxel_sizes, dim=0).reshape(-1)

    def print_summary(self):
        print("Current Octree Status:")
        print("Current depth:", self.cur_depth)
        total_nodes = 0
        for d in range(self.cur_depth + 1):
            num_nodes = self.nnum[d].item()
            total_nodes += num_nodes
        print("Total nodes:", total_nodes)

        total_active_nodes = 0
        for d in range(self.cur_depth + 1):
            active_mask = (self.children[d] == -1) & (self.weights[d] >= 1).any(dim=-1)
            total_active_nodes += active_mask.sum().item()
        print("Total active nodes with weights >= 1:", total_active_nodes)

        # NOTE: intentionally skipping fetch_active_points() here -- it concatenates
        # all per-depth leaves into 4 giant tensors and roughly doubles peak GPU
        # during the concat, which can dwarf the real integrate/render peaks we
        # want to measure. Static buffer + per-depth leaf count is enough.

        # check static data memory usage
        total_static_memory_usage = 0
        for d in range(self.cur_depth + 1):
            total_static_memory_usage += self.features[d].numel() * self.features[d].element_size()
            total_static_memory_usage += self.weights[d].numel() * self.weights[d].element_size()
            total_static_memory_usage += self.xyz[d].numel() * self.xyz[d].element_size()
            total_static_memory_usage += self.children[d].numel() * self.children[d].element_size()
            # if self.parent[d] is not None:
            #     total_static_memory_usage += self.parent[d].numel() * self.parent[d].element_size()
        print("Total memory usage of static octree data:", total_static_memory_usage / (1024 ** 2), "MB")
            
    def fetch_active_nodes_by_frustum(
        self,
        plane_normals: torch.Tensor,
        plane_offsets: torch.Tensor,
        start_depth: Optional[int] = None,
        weight_threshold: float = 0.0,
    ):
        """
        Fetch active leaf nodes that intersect with any of the given camera frustums.
        
        Starting from the full_depth layer (or `start_depth` if provided), checks
        cubes against the frustum planes:
        - Leaf nodes in frustum: added to update list
        - Non-leaf nodes in frustum: children are allocated and traversal continues deeper
        
        Args:
            plane_normals: (B, 4, 3) frustum plane normals in world space
            plane_offsets: (B, 4) plane offsets
            start_depth: Optional override for the depth level at which to begin
                traversal. Nodes at depths shallower than this are skipped entirely.
                Clamped to be at least `full_depth`. If None, defaults to `full_depth`.
            weight_threshold: Leaf nodes whose 8 corner weights are all strictly
                below this threshold are dropped from the returned indices. Matches
                the convention used by ``fetch_active_points``. Default 0.0 keeps
                every leaf in the frustum.
            
        Returns:
            per_depth_indices: List[Tensor], one per depth, containing the node indices
                of active leaf nodes within the frustum.
        """
        device = self.device
        per_depth_indices = [torch.tensor([], device=device, dtype=torch.long) for _ in range(self.cur_depth + 1)]
        
        # Start from full_depth level (or the user-provided override) - these nodes
        # are guaranteed to be allocated as long as they fit within cur_depth.
        if start_depth is None:
            start_depth = self.full_depth
        else:
            start_depth = max(int(start_depth), self.full_depth)
            start_depth = min(start_depth, self.cur_depth)
        
        # Get all node indices at start_depth
        num_nodes = self.nnum[start_depth].item()
        if num_nodes == 0:
            return per_depth_indices
            
        candidate_indices = torch.arange(num_nodes, device=device, dtype=torch.long)
        
        for d in range(start_depth, self.cur_depth + 1):
            if len(candidate_indices) == 0:
                break
                
            # Compute AABB centers and extents for candidate nodes at this depth
            xyz = self.xyz[d][candidate_indices]  # (M, 3)
            voxel_size = self.get_voxel_size(d)
            
            # Convert xyz grid coords to world-space centers
            # xyz_to_points gives (N, 8, 3) cube corners; center is midpoint
            centers = xyz.float() / (2 ** (d - 1)) - 1.0 + voxel_size * 0.5  # (M, 3)
            extents = torch.full_like(centers, voxel_size * 0.5)  # (M, 3)
            
            # Test frustum intersection
            inside_mask = frustum_aabb_intersection(centers, extents, plane_normals, plane_offsets)
            
            # Filter to only nodes inside frustum
            inside_indices = candidate_indices[inside_mask]
            
            if len(inside_indices) == 0:
                break
            
            # Separate leaf vs non-leaf nodes
            is_leaf = self.children[d][inside_indices] == -1
            
            # Leaf nodes go to update list
            leaf_indices = inside_indices[is_leaf]
            if len(leaf_indices) > 0:
                if weight_threshold > 0.0:
                    weight_mask = (self.weights[d][leaf_indices] >= weight_threshold).any(dim=-1)
                    leaf_indices = leaf_indices[weight_mask]
                if len(leaf_indices) > 0:
                    per_depth_indices[d] = torch.cat([per_depth_indices[d], leaf_indices])
            
            # Non-leaf nodes: traverse to children
            non_leaf_indices = inside_indices[~is_leaf]
            if len(non_leaf_indices) == 0 or d >= self.cur_depth:
                break
                
            # Allocate children for non-leaf nodes that don't have them yet
            needs_alloc = non_leaf_indices[self.children[d][non_leaf_indices] == -1]
            if len(needs_alloc) > 0:
                new_nodes = self.allocate_children(needs_alloc, d)
                if new_nodes.numel() > 0:
                    per_depth_indices[d + 1] = torch.cat([per_depth_indices[d + 1], new_nodes])
            
            # Gather all children of non-leaf nodes for next iteration
            child_base = self.children[d][non_leaf_indices]  # already have children
            valid_child_mask = child_base >= 0
            child_base = child_base[valid_child_mask]
            
            if len(child_base) == 0:
                break
            
            # Each parent has 8 children
            octant_offsets = torch.arange(8, device=device, dtype=torch.long)
            candidate_indices = (child_base.unsqueeze(1) << 3) + octant_offsets.unsqueeze(0)  # (P, 8)
            candidate_indices = candidate_indices.reshape(-1)
            
            # Validate indices
            next_num = self.nnum[d + 1].item()
            candidate_indices = candidate_indices[candidate_indices < next_num]
        
        # Deduplicate
        for d in range(len(per_depth_indices)):
            if len(per_depth_indices[d]) > 0:
                per_depth_indices[d] = torch.unique(per_depth_indices[d])
        
        return per_depth_indices
            