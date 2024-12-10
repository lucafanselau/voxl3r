from typing import Optional
from jaxtyping import jaxtyped, Bool, Float
from torch import Tensor
from . import base

class Config(base.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

    def add_image(self, image: Float[Tensor, "H W 3"], transformation: base.Transformation, intrinsics: Float[Tensor, "3 3"]) -> None:
        self.add_images(image[None, ...], transformation[None, ...], intrinsics[None, ...])

    @jaxtyped
    def add_images(self, 
                   images: Float[Tensor, "B H W 3"], 
                   transformation: base.BatchedTransformation,
                   intrinsics: Float[Tensor, "B 3 3"]) -> None:
        pass

