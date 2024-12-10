from typing import Optional
from beartype import beartype
from jaxtyping import jaxtyped, Bool
from torch import Tensor
import torch
import trimesh

from . import base
import datasets

class Config(base.Config, datasets.scene.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

        self.dataset = datasets.scene.Dataset(config)

    def add_scene(self, scene_id: str) -> None:
        idx = self.dataset.get_index_from_scene(scene_id)
        dict = self.dataset[idx]

        self.add_mesh(dict["mesh"])

    @jaxtyped(typechecker=beartype())
    def add_mesh(self, mesh: trimesh.Trimesh, transformation: Optional[base.Transformation] = None) -> None:
        if transformation is None:
            transformation = torch.eye(4, 4)

        # mesh = mesh.transform(transformation)

        self.plotter.add_mesh(mesh, opacity=0.5)
