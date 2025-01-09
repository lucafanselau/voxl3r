from pathlib import Path
import torch
from torch import nn, Tensor
from typing import Union, TypedDict
from datasets.chunk import image, mast3r, occupancy_revised as occupancy
from datasets.transforms import images
from utils.config import BaseConfig


class PointBasedTransformConfig(BaseConfig):
    pass


InputDict = Union[mast3r.Output, image.Output, occupancy.Output]
class OutputDict(TypedDict):
    X: Tensor
    Y: Tensor

class PointBasedTransform(nn.Module):
    def __init__(self, config: PointBasedTransformConfig):
        super().__init__()
        self.config = config
        self.transformation_transform = images.StackTransformations()

    def __call__(self, data: InputDict) -> OutputDict:
        grid_size = torch.tensor(data["grid_size"])
        center = data["center"].clone().detach()
        pitch = data["resolution"]

        # load images from pairwise_predictions and associated transformations
        res_dict = {
            **data["pairwise_predictions"][0],
            **data["pairwise_predictions"][1],
        }
        pred = data["pairwise_predictions"]
        image_dict = {
            Path(key).name: value
            for key, value in zip(data["images"][0], data["images"][1])
        }

        # transform a single pair into a 3d voxel grid
        # 

        # first get all the 3d points and associated transformations
        keys = list(res_dict.keys())


        


        
        # get T_0w from data
        # this reads as from the images get the transformations, then the one for the first (0) image and of this the full transformation matrix
        T_0w = torch.tensor(data["images"][1][0]["T_cw"])


        H, W = images.shape[-2:]
        transformations, T_cw, K = self.transformation_transform(image_dict, new_shape=torch.Tensor((H, W)))

        breakpoint()
        