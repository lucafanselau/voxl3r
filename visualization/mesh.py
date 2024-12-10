from typing import Optional
from jaxtyping import jaxtyped, Bool
from torch import Tensor
import trimesh

from . import base

class Config(base.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

    @jaxtyped
    def add_mesh(self, mesh: trimesh.Trimesh, transformation: Optional[base.Transformation] = None) -> None:
        pass
