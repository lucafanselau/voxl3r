from collections import defaultdict
from pathlib import Path
from typing import Union
from einops import rearrange
from jaxtyping import Float
from torch import nn
import torch

from datasets.transforms import images

from .projection_batched import batch_project_voxel_grid_to_images_seperate
from utils.config import BaseConfig


from datasets.chunk import mast3r, image, occupancy_revised as occupancy, image_loader

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
        images: Float[torch.Tensor, "B I C H W"],
        transformations: Float[torch.Tensor, "B I 3 4"],
        T_cw: Float[torch.Tensor, "B I 4 4"],
        coordinates: Float[torch.Tensor, "B F X Z Y"]
    ) -> Float[torch.Tensor, "B I C X Z Y"]:
        """
        Smear images into a feature grid

        B: Batch size
        I: Number of images (pairs are in the same dimension eg. 4 * 2)
        C: Number of channels
        H: Height
        W: Width

        F: Dimensions of the spatial feature grid
        """ 

        # Transform images into space
        input_grid, projected_depth, validity_indicator, viewing_direction = batch_project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
            grid_sampling_mode=self.config.grid_sampling_mode
        )

        if self.config.add_projected_depth:
            input_grid = torch.cat([input_grid, projected_depth], axis= -4)

        if self.config.add_validity_indicator:
            input_grid = torch.cat([input_grid, validity_indicator], axis = -4)

        if self.config.add_viewing_directing:
            input_grid = torch.cat([input_grid, viewing_direction], axis = -4)     
                
        return input_grid, coordinates
    


class BatchedStackTransformations(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.zeros(1)

    def forward(self, all_camera_params: dict[str, dict], new_shape: Float[torch.Tensor, "C 2"]):
        
        T_cw = torch.stack(
            [
            torch.stack([camera_params[key]["T_cw"] if torch.is_tensor(camera_params[key]["T_cw"]) else torch.from_numpy(camera_params[key]["T_cw"]).float()
                for key in camera_params.keys()
            ])
            for camera_params in all_camera_params.values()
            ]).to(self.tensor)
        
        K = defaultdict(dict)
        
        for camera_name, camera_params in all_camera_params.items():
            for key in camera_params.keys():
                K[camera_name][key] = camera_params[key]["K"] if torch.is_tensor(camera_params[key]["K"]) else torch.from_numpy(camera_params[key]["K"]).float()

        # now let's apply all of the new shapes 
        for i, (camera_name, camera_params) in enumerate(all_camera_params.items()):
            for key in camera_params.keys():                
                K[camera_name][key] = torch.from_numpy(mast3r.update_camera_intrinsics(K[camera_name][key].numpy(), new_shape[i]))

        transformation = torch.stack(
            [
                torch.stack([
                    K[camera_name][key] @ camera_params[key]["T_cw"][:3, :] if torch.is_tensor(camera_params[key]["T_cw"]) else torch.from_numpy(
                        K[camera_name][key].numpy() @ camera_params[key]["T_cw"][:3, :]
                    ).float()
                    for key in camera_params.keys()
                ])
                for camera_name, camera_params in all_camera_params.items()
            ]
        ).to(self.tensor)
        return transformation, T_cw, K

class SmearImagesConfig(BaseSmearConfig):
    smear_images_verbose: bool = False
    def get_feature_channels(self):
        return (2) * (3 + self.add_projected_depth + self.add_validity_indicator + self.add_viewing_directing * 3) 

SmearImagesDict = Union[image_loader.Output, occupancy.Output]

class SmearImages(BaseSmear):
    def __init__(self, config: SmearImagesConfig, *_args):
        super().__init__(config)
        self.config = config

        self.transformation_transform = BatchedStackTransformations()

    def __call__(self, batch) -> dict:
        # elements[0].keys() == dict_keys(['scene_name', 'center', 'image_name_chunk', 'images', 'cameras', 'file_name', 'pairs_indices', 'pairs_image_names', 'pairwise_prediction', 'coordinates', 'grid_size', 'resolution', 'T_random'])
        # only need to smear the images
        assert batch["type"] == "images", "SmearImages only works with image input"

        images = batch["X"]

        coordinates = batch["coordinates"]

        paired_transformations = batch["paired"]["transformations"]
        paired_T_cw = batch["paired"]["T_cw"]

        # first off, we need to batch the images together (currently they look like torch.Size([4, 2, 3, 1168, 1752]))
        transformations = rearrange(paired_transformations, "B I P ... -> B (I P) ...", P=2)
        T_cw = rearrange(paired_T_cw, "B I P ... -> B (I P) ...", P=2)
        
        images = rearrange(images, "B I P ... -> B (I P) ...", P=2)
        sampled, coordinates = self.smear_images(images, transformations, T_cw, coordinates=coordinates)
        
        # data["coordinates"] = coordinates
        sampled = rearrange(sampled, "B (I P) ... -> B I P ...", P=2)
        
        # now just return a dict that is compatible with the SmearMast3r output
        result = {
            "X": sampled,
            "Y": batch["Y"],
            # "images": [data["pairs_image_names"] for data in elements],
            # "scene_name": data["scene_name"],
            "coordinates": coordinates,
        }
        
        # if "occupancy_grid" in data.keys():
        #     result["Y"] = data["occupancy_grid"].bool().detach()
         
        if self.config.smear_images_verbose:
            result["verbose"] = {
                "batch": batch,
            }

        return result
    

##! NOTE: Mast3r is not currently ported to a batched transform


# class SmearMast3rConfig(BaseSmearConfig):
#     add_confidences: bool = False
#     add_pts3d: bool = False
#     ouput_only_training: bool = False
#     mast3r_verbose: bool = False

#     def get_feature_channels(self):
#         return (
#             (
#                 24
#                 + self.add_confidences
#                 + self.add_projected_depth
#                 + self.add_validity_indicator
#                 + self.add_viewing_directing * 3
#                 + self.add_pts3d * 3
#             )
#             * (2)
#         )

# SmearMast3rDict = Union[mast3r.Output, image.Output, occupancy.Output]

# class SmearMast3r(BaseSmear):
#     def __init__(self, config: SmearMast3rConfig, *_args):
#         super().__init__(config)
#         self.config = config
#         self.transformation_transform = images.StackTransformations()

#     @jaxtyped(typechecker=beartype)
#     def __call__(self, data: SmearMast3rDict) -> dict:

#         grid_size = torch.tensor(data["grid_size"])
#         center = data["center"].clone().detach()
#         pitch = data["resolution"]

#         # load images from pairwise_predictions and associated transformations
#         res1_dict = data["pairwise_predictions"][0]
#         res2_dict = data["pairwise_predictions"][1]
        
#         pairs_image_names = data["pairs_image_names"]
        
#         image_dict = {
#             Path(key).name: value
#             for key, value in zip(data["images"], data["cameras"])
#         }
        
#         # get T_0w from data
#         # this reads as from the images get the transformations, then the one for the first (0) image and of this the full transformation matrix
#         T_0w = torch.tensor(image_dict[pairs_image_names[0][0]]["T_cw"])
        
#         if self.config.add_confidences and self.config.add_pts3d:
#             images = torch.stack(
#                 [torch.cat([res1_dict["desc"], res1_dict["desc_conf"].unsqueeze(-1), res1_dict["pts3d"]], dim=-1), torch.cat([res2_dict["desc"], res2_dict["desc_conf"].unsqueeze(-1), res2_dict["pts3d"]], dim=-1)]
#             )
#         elif self.config.add_confidences:
#             images = torch.stack(
#                 [torch.cat([res1_dict["desc"], res1_dict["desc_conf"].unsqueeze(-1)], dim=-1), torch.cat([res2_dict["desc"], res2_dict["desc_conf"].unsqueeze(-1)], dim=-1)]
#             )
#         elif self.config.add_pts3d:
#             images = torch.stack(
#                 [torch.cat([res1_dict["desc"], res1_dict["pts3d"]], dim=-1), torch.cat([res2_dict["desc"], res2_dict["pts3d"]], dim=-1)]
#             )
#         else:
#             images = torch.stack(
#                 [res1_dict["desc"], res2_dict["desc"]]
#             ) 
            
        
#         images = rearrange(images, "P I H W C -> (I P) C H W", P = 2)
#         H, W = images.shape[-2:]
#         transformations, T_cw, K = self.transformation_transform(image_dict, new_shape=torch.Tensor((H, W)))
        
#         image_names = [Path(key).name for key in data["images"]]
#         pairs_idxs = [image_names.index(ele)  for pair in pairs_image_names for ele in pair]
        
#         # reorder transformations and T_cw to match order or images
#         transformations = transformations[pairs_idxs]
#         T_cw = T_cw[pairs_idxs]
        
#         sampled, coordinates = self.smear_images(grid_size, T_0w, center, pitch, images, transformations, T_cw)
#         sampled = rearrange(sampled, "(I P) ... -> I P ...", P=2)

#         # now just return a dict that is compatible with the SmearMast3r output
#         result = {
#             "X": sampled.detach(),
#             "Y": data["occupancy_grid"].bool().detach(),
#             "images": data["pairs_image_names"],
#         }
        
#         if self.config.mast3r_verbose:
#             result["verbose"] = {
#                 "coordinates" : coordinates,
#                 "data_dict" : data,
#                 "images": images,
#                 "add_confidences": self.config.add_confidences,
#                 "T_cw": T_cw,
#                 "K": K,
#                 "height" : H,
#                 "width" : W,
#             }

#         del data

#         return result