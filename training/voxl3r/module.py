


import torch
from torch import Tensor
from jaxtyping import Float
from networks.u_net import UNet3D
from networks.volume_transformer import VolumeTransformer, VolumeTransformerConfig
from training.mast3r.module_unet3d import UNet3DLightningModule, Unet3DLightningModuleConfig

class Voxl3RConfig(Unet3DLightningModuleConfig, VolumeTransformerConfig):
    pass

class Voxl3R(UNet3DLightningModule):

    model: UNet3D

    def __init__(self, module_config: Voxl3RConfig):
        super().__init__(module_config)

        # Additionally initialize a volume transformer
        self.volume_transformer = VolumeTransformer(module_config)

    def forward(
        self,
        x: Float[Tensor, "batch channels depth height width"]
    ) -> Float[Tensor, "batch 1 depth height width"]:
        self.model.encoder_forward(x)
