from torch import nn

from .smear_images import *
from .mirror import *
from .point_transform import *
from .images import *
from .shuffle import *
from .pe import *
from .sample_coordinate_grid import *
from .sample_occ_grid import *
from .pairing import *
from utils.config import BaseConfig
class ComposeTransformConfig(BaseConfig):
    transforms: Optional[list]
    
    
transform_dict = {
    "SmearImages": SmearImages,
    "MirrorTransform": MirrorTransform,
    "Shuffle": Shuffle,
    "PositionalEncoding": PositionalEncoding,
    "SmearMast3r": SmearMast3r,
    "SampleCoordinateGrid": SampleCoordinateGrid,
    "SampleOccGrid": SampleOccGrid,
    "Pairing": PairingTransform
}

class ComposeTransforms(nn.Module):
    def __init__(self, config: ComposeTransformConfig):
        super().__init__()
        self.transforms = nn.ModuleList([transform_dict[transform](config) for transform in config.transforms])
        

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict