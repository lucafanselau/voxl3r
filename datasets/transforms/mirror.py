import random
from torch import nn
import torch

from utils.config import BaseConfig


class MirrowConfig(BaseConfig):
    enable_mirror: bool = True

class MirrowTransform(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        config: MirrowConfig,
    ):
        super().__init__()
        self.config = config
        
    def forward(self, data):
        if self.config.enable_mirror:
            axes_to_flip = []
            for axis in [-1, -2, -3]:
                if random.random() < 0.5:
                    axes_to_flip.append(axis)
            
            if axes_to_flip:
                data["X"] = torch.flip(data["X"], dims=axes_to_flip)
                data["Y"] = torch.flip(data["Y"], dims=axes_to_flip)
            return data
        else:
            return data
    
