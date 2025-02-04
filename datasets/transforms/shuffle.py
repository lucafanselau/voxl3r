from torch import nn
import torch

from datasets import scene

class Shuffle(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        config: scene.Config,
        *_args,
    ):
        super().__init__()
        self.config = config
        
    def __call__(self, data: dict) -> dict:
        if self.config.split == "train":
            rand_idx = torch.randperm(data["X"].shape[0])
            data["X"] = data["X"][rand_idx]
        return data
    