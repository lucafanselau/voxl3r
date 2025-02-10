import random
from torch import nn
import torch

from datasets import scene
from utils.config import BaseConfig




class MirrorTransform(nn.Module):
    """
    Randomely flips input and output grids
    """

    def __init__(
        self,
        config: scene.Config,
        *_args,
    ):
        super().__init__()
        self.config = config
        
    def forward(self, data):
        if self.training:
            axes_to_flip = []
            for axis in [-2, -3]:
                if random.random() < 0.5:
                    axes_to_flip.append(axis)
            
            if axes_to_flip:
                if "X" in data.keys():
                    data["X"] = torch.flip(data["X"], dims=axes_to_flip)
                if "Y" in data.keys():
                    data["Y"] = torch.flip(data["Y"], dims=axes_to_flip)
                if "coordinates" in data.keys():
                    data["coordinates"] = torch.flip(data["coordinates"], dims=axes_to_flip)
        return data

    
