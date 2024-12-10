from typing import Optional
from jaxtyping import jaxtyped, Bool
from torch import Tensor
from . import base

class Config(base.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

    @jaxtyped
    def add_occupancy(self, occupancy: Bool[Tensor, "1 X Y Z"], transformation: Optional[base.Transformation] = None) -> None:
        pass