from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.mast3r_module import Mast3rModuleConfig
from utils.config import BaseConfig


class MLPConfig(BaseConfig):
    in_dim: int
    hidden_dim: int
    out_dim: int

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Patch2PixelsConfig(Mast3rModuleConfig):
    image_shape: tuple[int, int] = (512, 336)
    patch_size: int = 16

    def get_paired_pixel_dim(self):
        return (self.get_patch_emb_dim() // (self.patch_size ** 2)) * 2
    

class Patch2Pixels(nn.Module):
    """
    A PyTorch module implementing the core 3D SurfaceNet architecture
    based on Table 1 and Figure 3 of the paper.
    """
    def __init__(self, config: Patch2PixelsConfig):
       super().__init__()
       self.config = config
       mlp_config = MLPConfig(in_dim=config.get_patch_emb_dim(), hidden_dim=config.get_patch_emb_dim(), out_dim=config.get_patch_emb_dim())
       self.projector = MLP(mlp_config)  # B,S,D
       

    def forward(self, data_dict):

        assert data_dict["type"] == "patch", "Patch2Pixels only works with patch input"
        
        
        W, H = self.config.image_shape
        B, I, _2, S, E = data_dict["X"].shape
        data_dict["X"] = rearrange(data_dict["X"], "B I P ... -> (B I P) ...")
        projected_emb = self.projector(data_dict["X"])
        projected_emb = projected_emb.transpose(-1, -2).view((B * I * _2), -1, H // self.config.patch_size, W // self.config.patch_size)
        projected_images = F.pixel_shuffle(projected_emb, self.config.patch_size)
        
        data_dict["X"] = rearrange(projected_images, "(B I P) ... -> B I P ...", B=B, I=I, P=_2)
        data_dict["type"] = "images"
        return data_dict