from typing import Optional
from beartype import beartype
from jaxtyping import jaxtyped, Bool
import numpy as np
from torch import Tensor
import torch
import trimesh

from utils.chunking import chunk_mesh, mesh_2_local_voxels
from utils.transformations import invert_pose

from . import base
import datasets

class Config(base.Config, datasets.scene.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

        self.dataset = datasets.scene.Dataset(config)

    def add_scene(self, scene_id: str, opacity: Optional[float] = 0.5) -> None:
        idx = self.dataset.get_index_from_scene(scene_id)
        dict = self.dataset[idx]

        self.add_mesh(dict["mesh"], opacity=opacity)
        
    def add_chunked_mesh_from_zip_dict(self, chunk_dict: dict, opacity: Optional[float] = 0.5) -> None:
        idx = self.dataset.get_index_from_scene(chunk_dict["scene_name"])
        mesh = self.dataset[idx]["mesh"]
        
        image_dict = chunk_dict["images"][1][0]
        
        grid_size = np.array(chunk_dict["grid_size"])
        size = grid_size * chunk_dict["resolution"]
        mesh_chunked, backtransformed_mesh_chunked = chunk_mesh(
            mesh.copy(), image_dict["T_cw"], chunk_dict["center"], size, with_backtransform=True
        )

        self.add_mesh(backtransformed_mesh_chunked, opacity=opacity)

    @jaxtyped(typechecker=beartype)
    def add_mesh(self, mesh: trimesh.Trimesh, transformation: Optional[base.Transformation] = None, opacity: Optional[float] = 0.5) -> None:
        if transformation is None:
            transformation = torch.eye(4, 4)

        # mesh = mesh.transform(transformation)

        self.plotter.add_mesh(mesh, opacity=opacity)
