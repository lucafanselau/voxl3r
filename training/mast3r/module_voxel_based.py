from einops import rearrange
import torch
from networks.u_net import UNet3D, UNet3DConfig
from networks.voxel_based import VoxelBasedNetwork, VoxelBasedNetworkConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

import torch.nn as nn


class VoxelBasedLightningModuleConfig(UNet3DConfig, BaseLightningModuleConfig):
    pass


class VoxelBasedLightningModule(BaseLightningModule):
    def __init__(self, module_config: VoxelBasedNetworkConfig):
        super().__init__(module_config, VoxelBasedNetwork, module_config)
        
    def _shared_step(self, batch, batch_idx):
        
        x, y = batch["X"], batch["Y"]
                
        y_hat = self(x)
        
        y_hat = rearrange(y_hat, "B S W H D -> (B S) 1 W H D")
        y = rearrange(y, "B S W H D -> (B S) 1 W H D")
        
        loss = torch.tensor(0.0, device=self.device)
                
        N, C, W, H, D = y.shape
            
        count_pos = y.sum(dim=(1, 2, 3, 4))
        count_pos[count_pos == 0] = 1
        if (W * H * D < count_pos).sum():
                raise ValueError("The number of positive voxels is greater than the total number of voxels.")
        pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(y_hat, y.float())
        
        return loss, y_hat, y
       
