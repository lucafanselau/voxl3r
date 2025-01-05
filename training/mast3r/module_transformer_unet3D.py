from einops import rearrange
import torch
from networks.u_net import UNet3D, UNet3DConfig
from networks.volume_transformer import VolumeTransformerConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

import torch.nn as nn

from training.mast3r.module_unet3d import UNet3DLightningModule, Unet3DLightningModuleConfig


class TransformerUNet3LightningModuleConfig(VolumeTransformerConfig, Unet3DLightningModuleConfig):
    pass


class TransformerUNet3DLightningModule(VolumeTransformerConfig, UNet3DLightningModule):
    def __init__(self, module_config: Unet3DLightningModuleConfig):
        super().__init__(module_config, UNet3D, module_config)
       
