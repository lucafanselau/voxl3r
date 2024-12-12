
from pathlib import Path
from typing import Optional

from jaxtyping import Int, jaxtyped, Float
import torch
from torch import nn
from torchvision.io import read_image

from datasets.chunk.mast3r import update_camera_intrinsics
from extern.mast3r.dust3r.dust3r.utils.image import load_images

class StackTransformations(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.zeros(1)

    def forward(self, camera_params: dict, new_shape: Optional[Float[torch.Tensor, "2"]] = None):
        
        T_cw = torch.stack(
            [
            camera_params[key]["T_cw"] if torch.is_tensor(camera_params[key]["T_cw"]) else torch.from_numpy(camera_params[key]["T_cw"]).float()
                for key in camera_params.keys()
            ]).to(self.tensor)
        
        K = {}
        for key in camera_params.keys():
            K[key] = camera_params[key]["K"] if torch.is_tensor(camera_params[key]["K"]) else torch.from_numpy(camera_params[key]["K"]).float()
       
        if new_shape is not None:
            for key in camera_params.keys():                
                K[key] = torch.from_numpy(update_camera_intrinsics(K[key].numpy(), new_shape))

        transformation = torch.stack(
            [
                K[key] @ camera_params[key]["T_cw"][:3, :] if torch.is_tensor(camera_params[key]["T_cw"]) else torch.from_numpy(
                    K[key].numpy() @ camera_params[key]["T_cw"][:3, :]
                ).float()
                for key in camera_params.keys()
            ]
        ).to(self.tensor)
        return transformation, T_cw, K


class LoadImages(nn.Module):
    def __init__(self, data_dir: str):
        super().__init__()
        self.tensor = torch.zeros(1)
        self.data_dir = data_dir

    def forward(self, image_names: list[str]):
        """
        data dict is the data dict returned by create_chunk
        """

        images_dir = [str(Path(self.data_dir) / Path(*Path(image_name).parts[Path(image_name).parts.index("data") + 3 :])) for image_name in image_names]
        images_dir = [str(Path(self.data_dir) / Path(*Path(image_name).parts[Path(image_name).parts.index("data") + 3 :])) for image_name in images_dir]

        images = torch.stack([read_image(image_dir) for image_dir in images_dir]).to(
            self.tensor
        )
        return images