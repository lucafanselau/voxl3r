from typing import Optional
from torch import nn

from utils.config import BaseConfig

from torch.utils.data import default_collate


from .sample_occ_grid import *

class ComposeTransformConfig(BaseConfig):
    transforms_batched: Optional[list]
    
    
transform_dict = {
    "SampleOccGrid": SampleOccGrid,
}


# This is actually just a collate function
class ComposeTransforms(nn.Module):
    def __init__(self, config: ComposeTransformConfig):
        super().__init__()
        self.transforms = [transform_dict[transform](config) for transform in config.transforms_batched]
        

    def __call__(self, elements: list[dict]):
        # logic here is a bit different
        for transform in self.transforms:
            data_dict = transform(elements)
            elements.update(data_dict)

        elements["type"] = "images"
        return elements