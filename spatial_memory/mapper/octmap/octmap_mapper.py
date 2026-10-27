"""
OmniScene: 统一的场景管理类，整合 RGBDScene 和 MapIntegrator 的功能
支持：初始化、TSDF 融合、mesh 提取、渲染
"""

import torch
import numpy as np
import trimesh
import utils3d
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from .octree.dynamic_octree import DynOctree
from .camera import PinholeCamera
from .raycast_carver import RayCast_Carver
from .tsdf_updater import TSDFUpdater
from .mcube.octree_mesh_extractor import extract_mesh_from_octree
from .frustum import build_frustum_planes
from ..base_mapper import BaseMapper
from ...representations import MeshRepresentation


@dataclass
class IntegrationConfig:
    """TSDF 融合配置"""
    full_depth: int = 5
    octree_depth: int = 16
    sdf_truncate_margin_range: Tuple[float, float] = (0, 0.1)
    sdf_truncation_margin_ratio: float = 5.0
    carve_margin_ratio: float = 5.0
    use_carve_cube: bool = True
    carve_octree_depth_strategy: str = "pixel_size_to_volume_size"
    carve_bigger_voxel_pixel_size_scale: float = 2.0
    interplation_mode: str = "nearest"  # "nearest" or "trilinear"
    carve_cube_extend_size: int = 2  # Legacy field; allocator uses its own default (2).
    # Both modes allocate new frames only; offline resets observations and
    # updates all cached frames. map_frames' explicit argument takes precedence.
    integration_mode: str = "online"
    frustum_batch_size: int = 20  # Number of camera views to batch when fusing


class OctMapMapper(BaseMapper):
    """
    统一的场景管理类，支持:
    - 从 frames, depths, extrinsics, intrinsics 初始化
    - 场景归一化
    - TSDF 融合 (使用 FlexOctree)
    - Mesh 提取
    - 点云/Mesh 渲染
    """
    
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        integration_config: Optional[IntegrationConfig] = None,
        init_scene_scale: float = 0.5,
        device: str = "cuda",
        verbose=False,
        dump_integration_path: Optional[str] = None,
        dump_steps: int = 0,
        render_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        初始化 OmniScene
        
        Args:
            resolution: 目标分辨率 (H, W)，用于 integrate 时 resize 深度/图像
            integration_config: TSDF 融合配置
            device: 计算设备
            render_resolution: 渲染分辨率 (H, W)。与 integrate 分辨率解耦，None 时
                默认与 resolution 一致。
        """
        self.device = device
        super().__init__(
            resolution=resolution,
            device=device,
            render_resolution=render_resolution,
        )
        self.integration_config = integration_config or IntegrationConfig()
        
        # 初始化原始数据
        self.frames = None
        self.depths = None
        self.extrinsics = None
        self.intrinsics = None
        
        # 归一化参数
        self.init_scene_scale = init_scene_scale
        self._normalized = False
        
        # MapIntegrator 组件 (延迟初始化)
        self._octree: Optional[DynOctree] = None
        self._allocator: Optional[RayCast_Carver] = None
        self._updater: Optional[TSDFUpdater] = None
        self._integrated = False
        
        # Mesh
        self.mesh: Optional[MeshRepresentation] = None
        self._mesh_extracted = False
        self.verbose = verbose

        self.dump_integration_path: Optional[Path] = Path(dump_integration_path) if dump_integration_path is not None else None
        self.dump_steps: int = int(dump_steps)
        self._updated_frame_count: int = 0

        # Per-frame shallowest allocated octree depth, aligned with self.frames
        # (frame index order). Only maintained in offline integration mode, where
        # every chunk re-updates ALL frames and therefore needs each frame's own
        # allocation start depth instead of a single global one. Entries are
        # bumped by +1 whenever the octree root is extended (extend_root deepens
        # every existing node by one level).
        self._min_octree_depth_list: List[int] = []

    def register_min_octree_depth_list(self, depths: List[int]):
        """Append per-frame shallowest allocated octree depths.

        Called once per allocated chunk in offline mode: for a chunk of ``k``
        frames whose allocation produced a shallowest octree depth ``a``, pass
        ``[a] * k`` so the list stays aligned one-entry-per-frame with
        ``self.frames``. A ``None`` entry (allocation produced nothing) falls
        back to the octree ``full_depth``, matching the traversal default.
        """
        full_depth = self.integration_config.full_depth
        self._min_octree_depth_list.extend(
            full_depth if d is None else int(d) for d in depths
        )

    def __len__(self):
        return self.frames.shape[0] if self.frames is not None else 0
    
    def _init_integrator(self):
        """初始化 TSDF 融合组件"""
        cfg = self.integration_config
        self._octree = DynOctree(
            depth=cfg.octree_depth, 
            full_depth=cfg.full_depth, 
            device=self.device,
            interplation_mode=cfg.interplation_mode,
            # carve_cube_extend_size=cfg.carve_cube_extend_size,
        )
        self._allocator = RayCast_Carver(
            self._octree, 
            carve_margin_ratio=cfg.carve_margin_ratio, 
            use_carve_cube=cfg.use_carve_cube,
            octree_depth_strategy=cfg.carve_octree_depth_strategy,
            bigger_voxel_pixel_size_scale=cfg.carve_bigger_voxel_pixel_size_scale,
        )
        self._updater = TSDFUpdater(
            self._octree,
            sdf_truncate_margin_range=cfg.sdf_truncate_margin_range,
            trunc_margin_ratio=cfg.sdf_truncation_margin_ratio,
            frustum_batch_size=cfg.frustum_batch_size,
        )
    
    @property
    def octree(self) -> Optional[DynOctree]:
        """获取 octree"""
        return self._octree
    
    def normalize_scene(self, frame_indices: Optional[List[int]] = None, scene_scale: float = 0.5):
        """
        将场景归一化到 [-scale, scale] 的单位立方体
        
        Args:
            frame_indices: 用于估计归一化参数的帧索引列表，None 则使用所有帧
            scene_scale: 归一化的目标尺度
        """
        if frame_indices is None:
            frame_indices = list(range(len(self)))
        depth_mask = self.depths[frame_indices] > 0
        global_pcs = utils3d.depth_map_to_point_map(
            self.depths[frame_indices], self.intrinsics[frame_indices], self.extrinsics[frame_indices]
        )
        valid_global_pcs = global_pcs[depth_mask]
        
        t_scale = (valid_global_pcs.max(dim=0).values + valid_global_pcs.min(dim=0).values) / 2.0
        s_scale = (valid_global_pcs.max(dim=0).values - valid_global_pcs.min(dim=0).values).max() / 2.0
        
        self._scale = scene_scale / s_scale
        self._transform = t_scale
        
    def check_boundary(self):
        """检查场景边界，返回边界框信息"""
        if self.frames is None:
            raise RuntimeError("场景数据未初始化，无法检查边界")
        
        depth_mask = self.depths > 0
        global_pcs = utils3d.depth_map_to_point_map(
            self.depths, self.intrinsics, self.extrinsics
        )
        valid_global_pcs = global_pcs[depth_mask]
        # apply relative transformation
        valid_global_pcs = (valid_global_pcs - self._transform) * self._scale
        
        min_bound = valid_global_pcs.min()
        max_bound = valid_global_pcs.max()
        if min_bound < -1.0 or max_bound > 1.0:
            return False
        return True
        
    def map_frames(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        integration_mode: Optional[str] = None,
        only_allocate: bool = False,
    ):
        """
        扩展场景数据
        
        Args:
            frames: RGB 图像张量 (N, H, W, 3)
            depths: 深度图张量 (N, H, W)
            extrinsics: 相机外参张量 (N, 4, 4)
            intrinsics: 相机内参张量 (N, 3, 3)
            integration_mode: "online" 或 "offline"，None 时回退到 config 中的
                ``integration_mode``。
                - online: 增量式，只对新帧做 allocate + update。
                                - offline: 保留 octree 结构，只为新增帧 allocate，清零 observation
                                    weights，再对全部缓存帧重新 update。不是 DA3 streaming 开关。
            only_allocate: 是否仅进行 allocation 不做 TSDF update
        """
        integration_mode = integration_mode or self.integration_config.integration_mode
        if integration_mode not in ("online", "offline"):
            raise ValueError(f"integration_mode must be 'online' or 'offline', got {integration_mode!r}")
        extend_num = frames.shape[0]
        
        # all cached frames should be on CPU to avoid GPU memory issues
        frames = frames.cpu()
        depths = depths.cpu()
        extrinsics = extrinsics.cpu()
        intrinsics = intrinsics.cpu()
        
        if self.frames is None:
            self.frames = frames
            self.depths = depths
            self.extrinsics = extrinsics
            self.intrinsics = intrinsics
            min_octree_depth = self.integrate(only_allocate=only_allocate)
            if integration_mode == "offline":
                # Record this first chunk's start depth so later offline chunks
                # (which re-update ALL frames) can reuse it per-frame.
                self.register_min_octree_depth_list([min_octree_depth] * extend_num)
            if self.verbose:
                self._octree.print_summary()
            return
            
        self.frames = torch.cat([self.frames, frames], dim=0)
        self.depths = torch.cat([self.depths, depths], dim=0)
        self.extrinsics = torch.cat([self.extrinsics, extrinsics], dim=0)
        self.intrinsics = torch.cat([self.intrinsics, intrinsics], dim=0)
        
        # Grow the octree root until every cached point fits in [-1, 1].
        while not self.check_boundary():
            if self.verbose:
                print("[OctMap] Extend scene to fit new frames...")
            self._octree.extend_root()
            self._scale *= 0.5
            # extend_root deepens every existing node by one level, so previously
            # recorded per-frame start depths shift by +1 to stay consistent.
            if self._min_octree_depth_list:
                self._min_octree_depth_list = [d + 1 for d in self._min_octree_depth_list]
        
        if integration_mode == "offline":
            # Allocator logic is unchanged from online: allocate ONLY the newly
            # appended frames (the octree structure accumulates in the persistent
            # tree, so incremental allocation yields the same structure as an
            # allocate-all pass). Then wipe accumulated observations and re-update
            # ALL frames from a clean slate.
            #
            # NB: allocating over ALL frames here (frame_indices=None) would make
            # the RayCast_Carver hold every accumulated frame's rays at once, so
            # its peak memory would scale with the total frame count instead of
            # the chunk size — that is what made offline OOM vs online.
            new_indices = list(range(len(self) - extend_num, len(self)))
            min_octree_depth = self.integrate(frame_indices=new_indices, only_allocate=True)
            # Record the newly allocated chunk's shallowest start depth per-frame
            # (computed AFTER any extend_root above, so already on the current
            # tree's depth scale).
            self.register_min_octree_depth_list([min_octree_depth] * extend_num)
            self._octree.reset_observations()
            self._updated_frame_count = 0
            if not only_allocate:
                # Re-update ALL frames; feed the per-frame start-depth list so the
                # updater picks each frustum batch's own shallowest allocated
                # depth instead of a single global one (which let coarse leaves be
                # fused into distant frames -> large-face artifacts).
                self.integrate(frame_indices=None, only_update=True,
                               start_depth_list=self._min_octree_depth_list)
            if self.verbose:
                self._octree.print_summary()
            return
        
        # online: only integrate the newly appended frames.
        self.integrate(frame_indices=list(range(len(self)-extend_num, len(self))), only_allocate=only_allocate)
        if self.verbose:
            self._octree.print_summary()
    
    @torch.no_grad()
    def integrate(self, frame_indices: Optional[List[int]] = None, only_allocate: bool = False, only_update: bool = False,
                  start_depth_list: Optional[List[int]] = None):
        """
        执行 TSDF 融合 (always view-frustum batched).

        Args:
            frame_indices: 要融合的帧索引列表，None 则融合所有帧
            only_allocate: 仅分配，不更新
            only_update: 仅更新，不分配
            start_depth_list: 可选的 per-frame 起始 octree 深度列表（长度需与
                本次融合的帧数一致，顺序对齐 ``frame_indices``）。提供时会传给
                updater，使其对每个 frustum batch 取该 batch 帧对应的最小起始
                深度；None 时回退为 allocator 本次返回的全局 ``min_octree_depth``。
                主要用于 offline 模式的 only_update 全量重融合。

        Returns:
            min_octree_depth: 本次 allocation 的最浅 octree 深度（int）；当
                ``only_update`` 时无 allocation，返回 None。
        """

        start_time = time.perf_counter()

        if self._octree is None:
            self._init_integrator()
            if not self._normalized:
                self.normalize_scene(scene_scale=self.init_scene_scale)
                self._normalized = True
            
        if frame_indices is None:
            frame_indices = list(range(len(self)))
            
        frames = self.frames[frame_indices]
        depths = self.depths[frame_indices]
        extrinsics = self.extrinsics[frame_indices]
        intrinsics = self.intrinsics[frame_indices]
        
        # apply relative transformation    
        depths = depths.clone() * self._scale
        c2w = torch.inverse(extrinsics.clone())
        c2w[:, :3, 3] = (c2w[:, :3, 3] - self._transform) * self._scale
        extrinsics = torch.inverse(c2w)
        
        cfg = self.integration_config
        dense_octree = (cfg.full_depth == cfg.octree_depth)
        
        # Build cameras on-device (extrinsics/intrinsics are tiny). Keep the heavy
        # RGB/depth tensors on CPU: the allocator registers them one frame at a
        # time (releasing each depth immediately) and the updater moves them to
        # GPU one frustum-batch at a time. This bounds raw-input GPU residency to
        # the batch size instead of the total number of (re-)integrated frames.
        cameras_list = [
            PinholeCamera(extrinsic.to(self.device), intrinsic.to(self.device))
            for extrinsic, intrinsic in zip(extrinsics, intrinsics)
        ]
        
        # 分配阶段 (非稠密 octree): move only the current frame's depth to GPU and
        # release it right after registering (register_frame keeps compact voxel
        # keys, not the depth map).
        if not dense_octree and not only_update:
            for depth, camera in zip(depths, cameras_list):
                depth_dev = depth.to(self.device)
                self._allocator.register_frame(depth_dev, camera)
                del depth_dev
        
        # 运行分配
        _use_cuda_mem = self.verbose and isinstance(self.device, str) and self.device.startswith("cuda")
        if not dense_octree and not only_update:
            if _use_cuda_mem:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            min_octree_depth = self._allocator.run()
            if _use_cuda_mem:
                torch.cuda.synchronize(self.device)
                alloc_peak = torch.cuda.max_memory_allocated(self.device)
                print(f"[OctMap][mem] allocator.run peak: {alloc_peak / (1024 ** 2):.1f} MB")
            # torch.cuda.empty_cache() # clean the intermediate memory used by allocator before update
        else:
            min_octree_depth = None
        
        # 更新阶段
        if not only_allocate and _use_cuda_mem:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        if not only_allocate:
            # Batched view-frustum update: only update leaves visible to the current
            # batch of cameras. Use the shallowest depth allocated for this batch as
            # the starting search depth, skipping coarser leaves the current frames
            # could not have produced. When a per-frame ``start_depth_list`` is
            # supplied (offline re-update over all frames), pass it through so the
            # updater derives each batch's own shallowest depth; otherwise fall
            # back to the single global ``min_octree_depth`` from this allocation.
            updater_start_depth = start_depth_list if start_depth_list is not None else min_octree_depth
            self._updater.run(
                frames, depths, cameras_list,
                start_depth=updater_start_depth,
            )
            self._updated_frame_count += len(frames)

        if not only_allocate and _use_cuda_mem:
            torch.cuda.synchronize(self.device)
            updater_peak = torch.cuda.max_memory_allocated(self.device)
            print(f"[OctMap][mem] updater run peak: {updater_peak / (1024 ** 2):.1f} MB")

        self._integrated = True
        self._mesh_extracted = False

        elapsed = time.perf_counter() - start_time
        if self.verbose:
            print(f"[OctMap] integrate time: {elapsed:.4f}s")

        return min_octree_depth

    def extract_mesh(
        self,
        isolevel: float = 0.0,
        filter_min_faces: Optional[int] = 1000
    ) -> trimesh.Trimesh:
        """
        从 octree 提取 mesh
        
        Args:
            isolevel: 等值面级别
            filter_min_faces: 过滤小于此面数的连通分量，None 表示不过滤
            
        Returns:
            提取的 mesh
        """
        if not self._integrated:
            raise RuntimeError("请先调用 integrate() 进行 TSDF 融合")

        start_time = time.perf_counter()

        vertices, faces, colors = extract_mesh_from_octree(
            self._octree, 
            isolevel=isolevel
        )
        
        mesh = MeshRepresentation(
            vertices=vertices.cpu().numpy(),
            faces=faces.cpu().numpy(),
            colors=colors.cpu().numpy(),
            filter_min_faces=filter_min_faces,
        )
        
        self.mesh = mesh
        self._mesh_extracted = True

        elapsed = time.perf_counter() - start_time
        if self.verbose:
            print(f"[OctMap] extract_mesh time: {elapsed:.4f}s")
        return mesh

    def extract_mesh_by_view(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        isolevel: float = 0.0,
        filter_min_faces: Optional[int] = 1000,
        in_normalized_space: bool = False,
        weight_threshold: float = 1.0,
    ) -> MeshRepresentation:
        """
        从 octree 中只提取给定视锥内的 mesh（view-based extraction）。

        相比 ``extract_mesh()`` 全场景 marching cubes，这里先用 frustum culling 拿到
        每层 octree 中落在给定视锥联合内部的节点，再对这部分节点跑 marching cubes，
        从而大幅减少 mesh 提取的显存和时间开销，适合长程 integrate 场景下做
        per-view 的可视化或 render。

        Args:
            extrinsics: (N, 4, 4) or (4, 4) 相机外参 (world-to-camera)。
            intrinsics: (N, 3, 3) or (3, 3) 相机内参。
            isolevel: marching cubes 等值面阈值。
            filter_min_faces: 过滤小连通分量的最小面数，与 ``extract_mesh`` 一致。
            in_normalized_space: 若为 False，``extrinsics`` 视为原始世界空间，会被
                ``_transform_extrinsics`` 转到归一化空间后再用于建 frustum；若为 True
                则直接使用。
            weight_threshold: 过滤掉 8 个角点 weight 全部小于该阈值的 leaf，避免把
                还没被 TSDF 更新到的 carve-only 节点喂给 marching cubes。与
                ``extract_mesh`` 默认 (>=1) 行为保持一致。

        Returns:
            ``MeshRepresentation``，仅包含视锥内的 mesh 片段。
        """
        if not self._integrated:
            raise RuntimeError("请先调用 integrate() 进行 TSDF 融合")

        _use_cuda_mem = self.verbose and torch.cuda.is_available()

        def _stage_begin():
            if _use_cuda_mem:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            return time.perf_counter()

        def _stage_end(label: str, t0: float, stats: dict):
            if _use_cuda_mem:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            else:
                peak = float("nan")
            stats[label] = (time.perf_counter() - t0, peak)

        stage_stats: dict = {}
        start_time = time.perf_counter()

        # ---- input prep ----
        t0 = _stage_begin()
        # Normalise inputs to (B, 4, 4) / (B, 3, 3) on the octree device.
        if not isinstance(extrinsics, torch.Tensor):
            extrinsics = torch.from_numpy(extrinsics).float()
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.from_numpy(intrinsics).float()
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)

        if not in_normalized_space:
            extrinsics = self._transform_extrinsics(extrinsics)

        extrinsics = extrinsics.to(self.device).float()
        intrinsics = intrinsics.to(self.device).float()
        _stage_end("input_prep", t0, stage_stats)

        # ---- frustum planes ----
        t0 = _stage_begin()
        plane_normals, plane_offsets = build_frustum_planes(intrinsics, extrinsics)
        _stage_end("build_frustum_planes", t0, stage_stats)

        # ---- frustum-culled node fetch ----
        t0 = _stage_begin()
        per_depth_indices = self._octree.fetch_active_nodes_by_frustum(
            plane_normals, plane_offsets,
            weight_threshold=weight_threshold,
        )
        del plane_normals, plane_offsets
        _stage_end("fetch_active_nodes_by_frustum", t0, stage_stats)

        # ---- marching cubes ----
        t0 = _stage_begin()
        vertices, faces, colors = extract_mesh_from_octree(
            self._octree,
            isolevel=isolevel,
            per_depth_indices=per_depth_indices,
            verbose=self.verbose,
        )
        _stage_end("extract_mesh_from_octree", t0, stage_stats)

        # ---- mesh wrap (host transfer + filtering) ----
        t0 = _stage_begin()
        mesh = MeshRepresentation(
            vertices=vertices.cpu().numpy(),
            faces=faces.cpu().numpy(),
            colors=colors.cpu().numpy(),
            filter_min_faces=filter_min_faces,
        )
        _stage_end("mesh_wrap", t0, stage_stats)

        elapsed = time.perf_counter() - start_time
        if self.verbose:
            print(f"[OctMap] extract_mesh_by_view time: {elapsed:.4f}s")
            for k, (dt, peak) in stage_stats.items():
                print(f"  [stage] {k:<32s} {dt*1000:8.1f} ms   peak {peak:8.1f} MB")
        return mesh

    def render(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        """
        Render partial observations.

        Behaves like ``BaseMapper.render`` when a full-scene representation has
        already been built via ``extract_mesh`` / ``get_representation``. If no
        cached representation exists but the scene has been integrated (typical
        when ``extend_frames(..., skip_get_representation=True)`` was used in a
        long-horizon integration loop), this falls back to a view-based mesh
        extraction: only the octree leaves intersecting the requested cameras'
        frustums are turned into a transient mesh, rendered, and discarded.
        """
        if self._rep is not None:
            return super().render(extrinsics, intrinsics)

        if not self._integrated:
            raise RuntimeError("No representation available. Call extend_frames() first.")

        # extract_mesh_by_view handles the original->normalized transform itself.
        view_mesh = self.extract_mesh_by_view(
            extrinsics, intrinsics, in_normalized_space=False,
        )

        norm_extrinsics = self._transform_extrinsics(extrinsics)
        render_dict = view_mesh.render(norm_extrinsics, intrinsics, self.render_resolution)
        render_dict["depth"] = render_dict["depth"] / self._scale
        return render_dict

    def get_representation(self) -> MeshRepresentation:
        if self._mesh_extracted:
            return self.mesh
        else:
            return self.extract_mesh()

