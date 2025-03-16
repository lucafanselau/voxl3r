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


from ..chunk import mast3r, image, occupancy_revised as occupancy, image_loader
from . import images

class BaseSmearConfig(BaseConfig):
    pe_enabled: bool = False

    add_projected_depth: bool = False
    add_validity_indicator: bool = False
    add_viewing_directing: bool = False
    
    grid_sampling_mode: str = "nearest"
    seq_len: int


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

    def smear_images_batched(
        self,
        images: Float[torch.Tensor, "B P 2 C H W"],
        # combination of extrinsics and intrinsics from world to pixel space
        transformations_pw: Float[torch.Tensor, "B P 3 4"],
        # expected to be in world coordinates
        coordinates: Float[torch.Tensor, "B 3 X Y Z"],
    ) -> Float[torch.Tensor, "F X Z Y"]:
        pass

    #@jaxtyped(typechecker=beartype)
    def smear_images(
        self,
        images: Float[torch.Tensor, "I C H W"],
        transformations: Float[torch.Tensor, "I 3 4"],
        T_cw: Float[torch.Tensor, "I 4 4"],
        coordinates: Float[torch.Tensor, "F X Z Y"] = None,
    ) -> Float[torch.Tensor, "F X Z Y"]:
        """
        Smear images into a feature grid
        
        grid_size: size of the grid to smear the images into
        T_0w: world to camera 0 (eg. the one of image_name_chunk / occupancy_grid) transformation
        """

        # Transform images into space
        input_grid, projected_depth, validity_indicator, viewing_direction = project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
            grid_sampling_mode=self.config.grid_sampling_mode
        )

        if self.config.add_projected_depth:
            input_grid = torch.cat([input_grid, projected_depth], axis = 1)

        if self.config.add_validity_indicator:
            input_grid = torch.cat([input_grid, validity_indicator], axis = 1)

        if self.config.add_viewing_directing:
            input_grid = torch.cat([input_grid, viewing_direction], axis = 1)     
                
        return input_grid, coordinates

class SmearImagesConfig(BaseSmearConfig):
    smear_images_verbose: bool = False
    def get_feature_channels(self):
        return (2) * (3 + self.add_projected_depth + self.add_validity_indicator + self.add_viewing_directing * 3) 

SmearImagesDict = Union[image_loader.Output, occupancy.Output]

class SmearImages(BaseSmear):
    def __init__(self, config: SmearImagesConfig, *_args):
        super().__init__(config)
        self.config = config

        self.transformation_transform = images.StackTransformations()

    def __call__(self, data: SmearImagesDict) -> dict:
        # data.keys == dict_keys(['scene_name', 'center', 'image_name_chunk', 'images', 'cameras', 'file_name', 'pairwise_prediction', "pairs_image_names", 'resolution', 'grid_size', 'occupancy_grid'])
        # only need to smear the images
        images = data["pairwise_prediction"]

        image_dict = {
            Path(key).name: value
            for key, value in zip(data["images"], data["cameras"])
        }
        
        H, W = images.shape[-2:]
        new_shape = torch.Tensor((H, W)) if (data["cameras"][0]["height"], data["cameras"][0]["width"]) != (H, W) else None
        transformations, T_cw, _ = self.transformation_transform(image_dict, new_shape=new_shape)

        pairs_indices = data["pairs_indices"]
        transformations = transformations[pairs_indices]
        T_cw = T_cw[pairs_indices]

        # first off, we need to batch the images together (currently they look like torch.Size([4, 2, 3, 1168, 1752]))
        transformations = rearrange(transformations, "I P THREE FOUR -> (I P) THREE FOUR", P=2)
        T_cw = rearrange(T_cw, "I P FOUR FOUR2 -> (I P) FOUR FOUR2", P=2)
        
        images = rearrange(images, "I P C H W -> (I P) C H W", P=2)
        images = images / 255.0
        sampled, coordinates = self.smear_images(transformations, T_cw, coordinates=data["coordinates"])
        
        data["coordinates"] = coordinates
        sampled = rearrange(sampled, "(I P) ... -> I P ...", P=2)
        
        # now just return a dict that is compatible with the SmearMast3r output
        result = {
            "X": sampled.detach(),
            "images": data["pairs_image_names"],
            "scene_name": data["scene_name"],
            "coordinates": coordinates,
        }
        
        if "occupancy_grid" in data.keys():
            result["Y"] = data["occupancy_grid"].bool().detach()
         
        if self.config.smear_images_verbose:
            result["verbose"] = {
                "data_dict" : data,
                "images": images,
            }

        return result


class SmearMast3rConfig(BaseSmearConfig):
    add_confidences: bool = False
    add_pts3d: bool = False
    ouput_only_training: bool = False
    mast3r_verbose: bool = False

    output_logger_meta: bool = True

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
            * (2)
        )

SmearMast3rDict = Union[mast3r.Output, image.Output, occupancy.Output]

class SmearMast3r(BaseSmear):
    def __init__(self, config: SmearMast3rConfig, *_args):
        super().__init__(config)
        self.config = config
        self.transformation_transform = images.StackTransformations()

    @jaxtyped(typechecker=beartype)
    def __call__(self, data: SmearMast3rDict) -> dict:

        # load images from pairwise_predictions and associated transformations
        res1_dict = data["pairwise_predictions"][0]
        res2_dict = data["pairwise_predictions"][1]
        
        pairs_image_names = [(paths[0].name, paths[1].name) for paths in data["pairs_image_paths"]]
        
        
        if self.config.add_confidences and self.config.add_pts3d:
            images = torch.stack(
                [torch.cat([res1_dict["desc"], res1_dict["desc_conf"].unsqueeze(-1), res1_dict["pts3d"]], dim=-1), torch.cat([res2_dict["desc"], res2_dict["desc_conf"].unsqueeze(-1), res2_dict["pts3d"]], dim=-1)]
            )
        elif self.config.add_confidences:
            images = torch.stack(
                [torch.cat([res1_dict["desc"], res1_dict["desc_conf"].unsqueeze(-1)], dim=-1), torch.cat([res2_dict["desc"], res2_dict["desc_conf"].unsqueeze(-1)], dim=-1)]
            )
        elif self.config.add_pts3d:
            images = torch.stack(
                [torch.cat([res1_dict["desc"], res1_dict["pts3d"]], dim=-1), torch.cat([res2_dict["desc"], res2_dict["pts3d"]], dim=-1)]
            )
        else:
            images = torch.stack(
                [res1_dict["desc"], res2_dict["desc"]]
            ) 
            
        
        images = rearrange(images, "P I H W C -> (I P) C H W", P = 2)
        H, W = images.shape[-2:]
        
        transformations, T_cw, K = self.transformation_transform(data["camera_params"], new_shape=torch.Tensor((H, W)))
        
        image_names = list(data["camera_params"].keys())
        pairs_idxs = [image_names.index(ele)  for pair in pairs_image_names for ele in pair]
        
        # reorder transformations and T_cw to match order or images
        transformations = transformations[pairs_idxs]
        T_cw = T_cw[pairs_idxs]
        
        coordinates_smearing = data["coordinates_fine"] if "coordinates_fine" in data.keys() else data["coordinates"]
        
        sampled, _ = self.smear_images(images, transformations, T_cw, coordinates=coordinates_smearing)
        sampled = rearrange(sampled, "(I P) ... -> I P ...", P=2)

        # now just return a dict that is compatible with the SmearMast3r output
        result = {
            "X": sampled.detach(),
            "images": [str(path) for paths in data["pairs_image_paths"] for path in paths],
            "scene_name": data["scene_name"],
            "coordinates": data["coordinates"],
        }
        
        if "coordinates_fine" in data.keys():
            result["coordinates_fine"] = data["coordinates_fine"] 
            
        if self.config.output_logger_meta:
            logger = {
                "T_cw": T_cw,
                "K": torch.stack([k for k in K.values()]),
                "image_size": (H, W),
                "pairs_idxs": torch.tensor(pairs_idxs).reshape(-1, 2),
                "image_paths": [str(path) for paths in data["pairs_image_paths"] for path in paths],
                "origin": data["chunk_center"],
                "pitch": data["verbose"]["resolution"],
            }
        

        if self.config.output_logger_meta:
            result["logger"] = logger

        
        if "occupancy_grid" in data.keys():
            result["Y"] = data["occupancy_grid"].bool().detach()
        
        if self.config.mast3r_verbose:
            result["verbose"] = {
                "coordinates" : data["coordinates"],
                "coordinate_fine" : data["coordinates_fine"] if "coordinates_fine" in data.keys() else None,
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