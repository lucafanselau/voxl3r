from einops import rearrange
import torch
from networks.u_net import UNet3D, UNet3DConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

import torch.nn as nn


class Unet3DLightningModuleConfig(UNet3DConfig, BaseLightningModuleConfig):
    pass


class UNet3DLightningModule(BaseLightningModule):
    def __init__(self, module_config: Unet3DLightningModuleConfig, ModelClass=UNet3D):
        super().__init__(module_config, ModelClass, module_config)

        self.pos_weights = None #14.1
        
    def _shared_step(self, batch, batch_idx):
        
        x, y = batch["X"], batch["Y"]
                
        y_hat = self(x)
        
        """
        if isinstance(y_hat, list):
            y_hat = [rearrange(target, "B S W H D -> (B S) 1 W H D") for target in y_hat]
        else:
            if self.config.num_pairs is not None:
                y_hat = rearrange(y_hat, "B W H D -> B 1 W H D")
            else:
                y_hat = rearrange(y_hat, "B S W H D -> (B S) 1 W H D")
        
        if self.config.num_pairs is not None:
            y = rearrange(y, "B S W H D -> B S W H D")[:, 0, :, :, :].unsqueeze(1)
        else:
            y = rearrange(y, "B S W H D -> (B S) 1 W H D")
        """
        

        # TODO: This must all go into loss module!
        N, P, C, W, H, D = y.shape
        # also unpack prediction into pairs again
        y_hat = rearrange(y_hat, "(B P) C W H D -> B P C W H D", P=P)
        
        count_pos = y.sum(dim=(1, 2, 3, 4, 5)).float().mean()
        
        # exponentially weighted average of pos weights
        if count_pos != 0:
            if self.pos_weights is None:
                self.pos_weights = ((W * H * D - count_pos) / count_pos)
            else:
                self.pos_weights = 0.99 * self.pos_weights + 0.01 * ((W * H * D - count_pos) / count_pos)
        pos_weight = self.pos_weights.reshape(1, 1, 1, 1, 1)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = nn.BCEWithLogitsLoss()
        
        # N, C, W, H, D = y.shape
        
        # count_pos = y.sum(dim=(1, 2, 3, 4))
        # count_pos[count_pos == 0] = 1
        # pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        loss = criterion(y_hat, y.float())
            
            
        #self.log("pos_weight", self.pos_weights.item(), on_step=True)
        self.log("pos_weight", pos_weight.mean().item(), on_step=True)
            
        return loss, y_hat[0] if isinstance(y_hat, list) else y_hat, y
       
