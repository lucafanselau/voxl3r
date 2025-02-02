from typing import Optional
from torch import nn

from utils.config import BaseConfig


from .sample_occ_grid import *

class ComposeTransformConfig(BaseConfig):
    transforms_batched: Optional[list]
    
    
transform_dict = {
    "SampleOccGrid": SampleOccGrid
}

class ComposeTransforms(nn.Module):
    def __init__(self, config: ComposeTransformConfig):
        super().__init__()
        self.transforms = [transform_dict[transform] for transform in config.transforms_batched]
        

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict