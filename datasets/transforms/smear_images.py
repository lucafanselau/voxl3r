from pathlib import Path
from typing import Optional, Tuple, TypedDict, Union
from beartype import beartype
from einops import rearrange
from jaxtyping import Int, jaxtyped, Float
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

    concatinate_pe: bool = False

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
        verbose: Optional[bool] = False,
    ) -> Float[torch.Tensor, "F X Z Y"]:
        """
        Smear images into a feature grid
        
        grid_size: size of the grid to smear the images into
        T_0w: world to camera 0 (eg. the one of image_name_chunk / occupancy_grid) transformation
        """

        seq_len = images.shape[0]

        # compute the coordinates of each point in shape
        _, _, T_w0 = invert_pose(T_0w[:3, :3], T_0w[:3, 3])
        coordinates = compute_coordinates(
            grid_size.numpy(),
            center.numpy(),
            pitch,
            grid_size[0].item(),
            to_world_coordinates=T_w0,
        )
        coordinates = torch.from_numpy(coordinates).float().to(grid_size.device)

        # Transform images into space
        feature_grid = project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
        )

        local_features, projected_depth, validity_indicator, viewing_direction = (
            feature_grid
        )

        rand_idx = torch.randperm(self.config.seq_len)
        indices_viewing_dir = (
            torch.arange(viewing_direction.shape[0])
            .reshape(-1, viewing_direction.shape[0] // seq_len)[rand_idx]
            .flatten()
        )
        indices_local_features = (
            torch.arange(local_features.shape[0])
            .reshape(-1, local_features.shape[0] // seq_len)[rand_idx]
            .flatten()
        )

        # shuffle the features in first dimension
        local_features = local_features[indices_local_features]
        viewing_direction = viewing_direction[indices_viewing_dir]
        projected_depth = projected_depth[rand_idx]
        validity_indicator = validity_indicator[rand_idx]

        sampled = local_features

        if self.config.add_projected_depth:
            sampled = torch.cat([sampled, projected_depth])

        if self.config.add_validity_indicator:
            sampled = torch.cat([sampled, validity_indicator])

        if self.config.add_viewing_directing:
            sampled = torch.cat([sampled, viewing_direction])

        fill_value = -1.0

        C, W, H, D = sampled.shape

        # TODO: make this dynamic

        num_of_channels = self.config.seq_len * (
            local_features.shape[0] // seq_len
            + self.config.add_projected_depth
            + self.config.add_validity_indicator
            + viewing_direction.shape[0] // seq_len * self.config.add_viewing_directing
        )

        if num_of_channels - C:
            sampled = torch.cat(
                [
                    sampled,
                    torch.full(
                        (num_of_channels - C, W, H, D),
                        fill_value,
                    ).to(sampled.device),
                ],
                dim=-1,
            )

        # give positional envoding even though values can be just filled with -1?
        if self.config.pe_enabled:
            channels = sampled.shape[0]

            if self.pe is None:
                self.pe = PositionalEncoding3D(channels).to(sampled.device)

            sampled_reshaped = rearrange(sampled, "C X Y Z -> 1 X Y Z C")
            pe_tensor = self.pe(sampled_reshaped)
            pe_tensor = rearrange(pe_tensor, "1 X Y Z C -> C X Y Z")
            if self.config.concatinate_pe:
                sampled = torch.cat([sampled, pe_tensor], dim=0)
            else:
                sampled = sampled + pe_tensor

        if verbose:
            return sampled, coordinates
        return sampled

class SmearMast3rConfig(BaseSmearConfig):
    add_confidences: bool = False
    ouput_only_training: bool = False
    mast3r_verbose: bool = False

    def get_feature_channels(self):
        return (
            (
                24
                + self.add_confidences
                + self.add_projected_depth
                + self.add_validity_indicator
                + self.add_viewing_directing * 3
            )
            * self.seq_len
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
        center = torch.tensor(data["center"])
        pitch = data["resolution"]

        # get T_0w from data
        # this reads as from the images get the transformations, then the one for the first (0) image and of this the full transformation matrix
        T_0w = torch.tensor(data["images"][1][0]["T_cw"])

        # load images from pairwise_predictions and associated transformations
        res_dict = {
            **data["pairwise_predictions"][0],
            **data["pairwise_predictions"][1],
        }
        image_dict = {
            Path(key).name: value
            for key, value in zip(data["images"][0], data["images"][1])
        }
        if self.config.add_confidences:
            images = torch.stack(
                [torch.cat([res_dict[f"desc_{image}"], res_dict[f"desc_conf_{image}"].unsqueeze(-1)], dim=-1) for image in image_dict.keys()]
            )
        else:
            images = torch.stack(
                [res_dict[f"desc_{image}"] for image in image_dict.keys()]
        ) 
        images = rearrange(images, "I H W C -> I C H W")
        H, W = images.shape[-2:]
        transformations, T_cw, K = self.transformation_transform(image_dict, new_shape=torch.Tensor((H, W)))


        if self.config.mast3r_verbose:
            sampled, coordinates = self.smear_images(grid_size, T_0w, center, pitch, images, transformations, T_cw, verbose = self.config.mast3r_verbose)
        else:
            sampled = self.smear_images(grid_size, T_0w, center, pitch, images, transformations, T_cw, verbose = self.config.mast3r_verbose)

        result = {}
        result["Y"] = data["occupancy_grid"].int().detach()
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