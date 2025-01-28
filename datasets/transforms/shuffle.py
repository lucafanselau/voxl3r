from torch import nn
import torch

class Shuffle(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        *_args,
    ):
        super().__init__()
        
    def __call__(self, data: dict) -> dict:
        rand_idx = torch.randperm(data["X"].shape[0])
        data["X"] = data["X"][rand_idx]
        return data
    