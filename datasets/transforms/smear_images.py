from pathlib import Path
from typing import Optional, Tuple, TypedDict, Union
from beartype import beartype
from einops import rearrange
from jaxtyping import Int, jaxtyped, Float
import numpy as np
from torch import nn
import torch

from .projection import project_voxel_grid_to_images_seperate
from utils.chunking import compute_coordinates
from utils.config import BaseConfig
from utils.transformations import invert_pose
from positional_encodings.torch_encodings import PositionalEncoding3D


from ..chunk import mast3r, image, occupancy
from . import images

class BaseSmearConfig(BaseConfig):
    pe_enabled: bool = False

    add_projected_depth: bool = False
    add_validity_indicator: bool = False
    add_viewing_directing: bool = False
    
    grid_sampling_mode: str = "nearest"

    concatinate_pe: bool = False
    shuffle_images: bool = False

    seq_len: int = 4


class BaseSmear(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        config: BaseSmearConfig,
    ):
        super().__init__()
        self.config = config
        self.pe = None

    #@jaxtyped(typechecker=beartype)
    def smear_images(
        self,
        grid_size: Int[torch.Tensor, "3"],
        T_0w: Float[torch.Tensor, "4 4"],
        center: Float[torch.Tensor, "3"],
        pitch: float,
        images: Float[torch.Tensor, "I C H W"],
        transformations: Float[torch.Tensor, "I 3 4"],
        T_cw: Float[torch.Tensor, "I 4 4"],
    ) -> Float[torch.Tensor, "F X Z Y"]:
        """
        Smear images into a feature grid
        
        grid_size: size of the grid to smear the images into
        T_0w: world to camera 0 (eg. the one of image_name_chunk / occupancy_grid) transformation
        """

        # compute the coordinates of each point in shape
        _, _, T_w0 = invert_pose(T_0w[:3, :3], T_0w[:3, 3])
        coordinates = torch.from_numpy(compute_coordinates(
            grid_size.numpy(),
            center.numpy(),
            pitch,
            grid_size[0].item(),
            to_world_coordinates=T_w0,
        )).float().to(grid_size.device)

        # Transform images into space
        input_grid, projected_depth, validity_indicator, viewing_direction = project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
            grid_sampling_mode=self.config.grid_sampling_mode
        )
        
        # if self.config.shuffle_images:
        #     rand_idx = torch.randperm(self.config.seq_len)
            
        #     input_grid = input_grid[rand_idx]
        #     viewing_direction = viewing_direction[rand_idx]
        #     projected_depth = projected_depth[rand_idx]
        #     validity_indicator = validity_indicator[rand_idx]
        
        # if self.config.shuffle_images:
        #     rand_idx = 2*torch.randperm(self.config.seq_len//2)
        #     rand_idx = [x.item() for r in rand_idx for x in (r, r+1)]
            
        #     input_grid = input_grid[rand_idx]
        #     viewing_direction = viewing_direction[rand_idx]
        #     projected_depth = projected_depth[rand_idx]
        #     validity_indicator = validity_indicator[rand_idx]
            
        if self.config.shuffle_images:
            rand_idx_0 = 2*torch.randperm(self.config.seq_len//2)
            rand_idx_1 = 2*torch.randperm(self.config.seq_len//2) + 1
            rand_idx = [x.item() for r1, r2 in zip(rand_idx_0, rand_idx_1) for x in (r1, r2)]
            
            input_grid = input_grid[rand_idx]
            viewing_direction = viewing_direction[rand_idx]
            projected_depth = projected_depth[rand_idx]
            validity_indicator = validity_indicator[rand_idx]


        if self.config.add_projected_depth:
            input_grid = torch.cat([input_grid, projected_depth], axis = 1)

        if self.config.add_validity_indicator:
            input_grid = torch.cat([input_grid, validity_indicator], axis = 1)

        if self.config.add_viewing_directing:
            input_grid = torch.cat([input_grid, viewing_direction], axis = 1)

        fill_value = -1.0

        I, C, W, H, D = input_grid.shape

        # TODO: make this dynamic / is this neccecary?
        num_of_channels = (
            input_grid.shape[1]
            + self.config.add_projected_depth
            + self.config.add_validity_indicator
            + viewing_direction.shape[1] * self.config.add_viewing_directing
        )

        if num_of_channels - C:
            input_grid = torch.cat(
                [
                    input_grid,
                    torch.full(
                        (I, num_of_channels - C, W, H, D),
                        fill_value,
                    ).to(input_grid.device),
                ],
                dim=-1,
            )

        if self.config.seperate_image_pairs is False:   
            input_grid = rearrange(input_grid, "P C X Y Z -> (P C) X Y Z")   
            channels = input_grid.shape[0] 
        else:
            num_of_pairs = self.config.seq_len // 2
            input_grid = rearrange(input_grid, "(num_of_pairs I) C X Y Z -> num_of_pairs (I C) X Y Z", I=2, num_of_pairs=num_of_pairs)
            channels = input_grid.shape[1]
                
        if self.config.pe_enabled:
                
            if self.pe is None:
                self.pe = PositionalEncoding3D(channels).to(input_grid.device)

            pe_tensor = self.pe(rearrange(input_grid, "P C X Y Z -> P X Y Z C" if self.config.seperate_image_pairs else "C X Y Z -> 1 X Y Z C"))
            
            if self.config.seperate_image_pairs is False: 
                pe_tensor = rearrange(pe_tensor, "1 X Y Z C -> C X Y Z")
            else:
                pe_tensor = rearrange(pe_tensor, "P X Y Z C -> P C X Y Z")
                
            if self.config.concatinate_pe:
                input_grid = torch.cat([input_grid, pe_tensor], dim=0)
            else:
                input_grid = input_grid + pe_tensor

        return input_grid, coordinates

class SmearMast3rConfig(BaseSmearConfig):
    add_confidences: bool = False
    add_pts3d: bool = False
    ouput_only_training: bool = False
    mast3r_verbose: bool = False
    seperate_image_pairs: bool = False

    def get_feature_channels(self):
        return (
            (
                24
                + self.add_confidences
                + self.add_projected_depth
                + self.add_validity_indicator
                + self.add_viewing_directing * 3
                + self.add_pts3d * 3
            )
            * (2 if self.seperate_image_pairs else self.seq_len)
            * (1 + self.concatinate_pe)
        )

SmearMast3rDict = Union[mast3r.Output, image.Output, occupancy.Output]

class SmearMast3r(BaseSmear):
    def __init__(self, config: SmearMast3rConfig):
        super().__init__(config)
        self.config = config
        self.transformation_transform = images.StackTransformations()

    @jaxtyped(typechecker=beartype)
    def __call__(self, data: SmearMast3rDict) -> dict:

        grid_size = torch.tensor(data["grid_size"])
        center = data["center"].clone().detach()
        pitch = data["resolution"]

        # load images from pairwise_predictions and associated transformations
        res_dict = {
            **data["pairwise_predictions"][0],
            **data["pairwise_predictions"][1],
        }
        image_dict = {
            Path(key).name: value
            for key, value in zip(data["images"][0], data["images"][1])
        }
        
        # get T_0w from data
        # this reads as from the images get the transformations, then the one for the first (0) image and of this the full transformation matrix
        T_0w = torch.tensor(data["images"][1][0]["T_cw"])
        
        if self.config.add_confidences and self.config.add_pts3d:
            images = torch.stack(
                [torch.cat([res_dict[f"desc_{image}"], res_dict[f"desc_conf_{image}"].unsqueeze(-1), res_dict[f"pts3d_{image}"]], dim=-1) for image in image_dict.keys()]
            )
        elif self.config.add_confidences:
            images = torch.stack(
                [torch.cat([res_dict[f"desc_{image}"], res_dict[f"desc_conf_{image}"].unsqueeze(-1)], dim=-1) for image in image_dict.keys()]
            )
        elif self.config.add_pts3d:
            images = torch.stack(
                [torch.cat([res_dict[f"desc_{image}"], res_dict[f"pts3d_{image}"]], dim=-1) for image in image_dict.keys()]
            )
        else:
            images = torch.stack(
                [res_dict[f"desc_{image}"] for image in image_dict.keys()]
        ) 
        images = rearrange(images, "I H W C -> I C H W")
        H, W = images.shape[-2:]
        transformations, T_cw, K = self.transformation_transform(image_dict, new_shape=torch.Tensor((H, W)))

        sampled, coordinates = self.smear_images(grid_size, T_0w, center, pitch, images, transformations, T_cw)

        result = {}
        
        occ = data["occupancy_grid"]
        if self.config.seperate_image_pairs:
            result["Y"] = occ.int().detach().repeat(sampled.shape[0], 1, 1, 1)
        else:
            result["Y"] = occ.int().detach()
        
        result["X"] = sampled.detach()
        
        if self.config.mast3r_verbose:
            result["verbose"] = {
                "coordinates" : coordinates,
                "data_dict" : data,
                "images": images,
                "add_confidences": self.config.add_confidences,
                "T_cw": T_cw,
                "K": K,
                "height" : H,
                "width" : W,
            }

        del data

        return result