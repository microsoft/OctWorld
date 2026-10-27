from trimesh import Trimesh
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch
import numpy as np
import trimesh
from trimesh import graph
from spatial_memory.renderers import render_point_cloud, render_mesh

class Representation(ABC):
    
    def render(self, extrinsics, intrinsics, resolution):
        if not isinstance(extrinsics, torch.Tensor):
            extrinsics = torch.from_numpy(extrinsics).float()
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.from_numpy(intrinsics).float()
        if extrinsics.ndim == 2 and intrinsics.ndim == 2:
            return self._render(extrinsics, intrinsics, resolution)
        else:
            assert extrinsics.ndim == 3 and intrinsics.ndim == 3 and extrinsics.shape[0] == intrinsics.shape[0], "extrinsics and intrinsics should have the same batch size"
            render_dict = {
                "color": [],
                "depth": []
            }
            for extrinsic, intrinsic in zip(extrinsics, intrinsics):
                render_result = self._render(extrinsic, intrinsic, resolution)
                render_dict["color"].append(render_result["color"])
                render_dict["depth"].append(render_result["depth"])
            render_dict["color"] = torch.stack(render_dict["color"], dim=0)
            render_dict["depth"] = torch.stack(render_dict["depth"], dim=0)
            return render_dict
    
    @abstractmethod
    def _render(self, extrinsics, intrinsics, resolution):
        # 根据representation的类型和相机参数去render partial observation
        pass
    
    @abstractmethod
    def dump_to_file(self, file_path):
        # dump representation to file, for example, save mesh to .ply or .obj, save point cloud to .ply
        pass
    
class MeshRepresentation(Representation):
    def __init__(self, vertices, faces, colors, filter_min_faces: Optional[int] = None):
        mesh = Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        
        if filter_min_faces is not None and filter_min_faces > 0:
            self.mesh = self.filter_mesh_by_faces(mesh, min_faces=filter_min_faces)
        else:
            self.mesh = mesh
    
    def _render(self, extrinsics, intrinsics, resolution):
        return render_mesh(self.mesh, resolution, extrinsics, intrinsics)
    
    @staticmethod
    def filter_mesh_by_faces(mesh: trimesh.Trimesh, min_faces: int = 1000):
        """根据面片数量筛选 mesh 的子组件"""
        components = graph.connected_components(
            edges=mesh.face_adjacency, 
            nodes=np.arange(len(mesh.faces)), 
            min_len=min_faces, 
            engine="scipy"
        )
        if len(components) == 0:
            return mesh
        preserve_face = np.unique(np.concatenate(components, axis=0))
        mesh.update_faces(preserve_face)
        mesh.remove_unreferenced_vertices()
        return mesh
    
    def dump_to_file(self, file_path):
        self.mesh.export(file_path)
        
class PointCloudRepresentation(Representation):
    def __init__(self, points, colors):
        self.points = points if isinstance(points, torch.Tensor) else torch.from_numpy(points).float()
        self.colors = colors if isinstance(colors, torch.Tensor) else torch.from_numpy(colors).float()
    
    def _render(self, extrinsics, intrinsics, resolution):
        return render_point_cloud(self.points, self.colors, resolution, extrinsics, intrinsics)
    
    def dump_to_file(self, file_path):
        cloud = trimesh.points.PointCloud(
            self.points.cpu().numpy(),
            colors=(self.colors.cpu().numpy() * 255).astype(np.uint8) if self.colors is not None else None
        )
        cloud.export(file_path)