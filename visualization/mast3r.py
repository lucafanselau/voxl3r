from typing import Optional
from jaxtyping import jaxtyped, Bool, Float
from torch import Tensor
from . import images
from . import base

class Config(images.Config):
    pass

class Visualizer(images.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)


    def add_mast3r_images(self, 
                   images: Float[Tensor, "B H W F"], 
                   transformation: base.BatchedTransformation,
                   intrinsics: Float[Tensor, "B 3 3"],
                   confidences: Optional[Float[Tensor, "B H W 1"]] = None
                   ) -> None:
        """
        Add a batch of images to the visualizer.

        Args:
            images: HAS TO BE WITHOUT CONFIDENCES!
        """
        pass

