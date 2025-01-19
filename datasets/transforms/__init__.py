from torch import nn


from .smear_images import *
#from .point_based import *
from .point_transform import *
from .images import *

class ComposeDictTransforms(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict