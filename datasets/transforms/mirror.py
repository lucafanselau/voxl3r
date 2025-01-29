import random
from torch import nn
import torch

from utils.config import BaseConfig


class MirrorTransform(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        *_args,
    ):
        super().__init__()
        
    def forward(self, data):
        axes_to_flip = []
        for axis in [-1, -2, -3]:
            if random.random() < 0.5:
                axes_to_flip.append(axis)
        
        if axes_to_flip:
            data["X"] = torch.flip(data["X"], dims=axes_to_flip)
            data["Y"] = torch.flip(data["Y"], dims=axes_to_flip)
        return data

    
