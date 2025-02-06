from typing import Optional
from torch import nn

from utils.config import BaseConfig


from .sample_occ_grid import *
from .smear import *

class ComposeTransformConfig(BaseConfig):
    transforms_batched: Optional[list]
    
    
transform_dict = {
    "SampleOccGrid": SampleOccGrid,
    "SmearImages": SmearImages
}

class ComposeTransforms(nn.Module):
    def __init__(self, config: ComposeTransformConfig):
        super().__init__()
        self.transforms = [transform_dict[transform] for transform in config.transforms_batched]
        

    def __call__(self, elements: list[dict]):
        # logic here is a bit different
        result = {}
        for transform in self.transforms:
            data_dict = transform(elements)
            result.update(data_dict)
        return result