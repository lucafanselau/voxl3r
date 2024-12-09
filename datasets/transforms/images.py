
from pathlib import Path
import torch
from torch import nn
from torchvision.io import read_image

class StackTransformations(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.zeros(1)

    def forward(self, camera_params: dict):
        T_cw = torch.stack(
            [
               cp["T_cw"] if torch.is_tensor(cp["T_cw"]) else torch.from_numpy(cp["T_cw"]).float()
                for cp in camera_params.values()
            ]
        ).to(self.tensor)
        transformation = torch.stack(
            [
                cp["K"] @ cp["T_cw"][:3, :] if torch.is_tensor(cp["T_cw"]) else torch.from_numpy(
                    cp["K"] @ cp["T_cw"][:3, :]
                ).float()
                for cp in camera_params.values()
            ]
        ).to(self.tensor)
        return transformation, T_cw


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