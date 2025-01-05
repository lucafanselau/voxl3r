#! Voxl3r is essentially the u_net.py network, but with a VIT-inspired sequential model in the latent space of the UNet.

from charset_normalizer import from_path
from . import volume_transformer
from . import u_net
from dataclasses import field

class Voxl3rConfig(u_net.UNet3DConfig):
    grid_size: list[int] = field(
        default_factory=lambda: [64, 64, 64]
    )
    vt_depth: int = 4 # number of layers in the volume transformer
    vt_heads: int = 8 # number of attention heads in the volume transformer
    vt_dim_head: int = 64 # number of dimensions in the attention head of the volume transformer
    vt_mlp_dim: int = 256 # number of dimensions in the feedforward layer of the volume transformer


class Voxl3r(u_net.UNet3D):
    def __init__(self, config: Voxl3rConfig):
        super().__init__(config)

        self.config = config

        # The SimpleViT model is programmed to use a "video"
        # we now want to transform the 3d voxel grids in the latent dimension to work "as a video"
        self.vit = volume_transformer.VolumeTransformer(
            image_size=tuple(self.config.grid_size),
            image_patch_size=1,
            dim=self.latent_dim,
            depth=self.config.vt_depth,
            heads=self.config.vt_heads,
            mlp_dim=self.config.vt_mlp_dim,
            dim_head=self.config.vt_dim_head,
        )

    @staticmethod
    def load_from_trained_unet(unet_path: str, config: Voxl3rConfig):
        unet = u_net.UNet3D.load_from_checkpoint(unet_path, config=config)
        vit = volume_transformer.VolumeTransformer(
            image_size=tuple(config.grid_size),
            image_patch_size=1,
            dim=unet.latent_dim,
            depth=config.vt_depth,
            heads=config.vt_heads,
            mlp_dim=config.vt_mlp_dim,
            dim_head=config.vt_dim_head,
        )
        return vit

        