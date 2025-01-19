from einops import rearrange
import torch
from networks.point_transformer import PointTransformer, PointTransformerConfig
from networks.u_net import UNet3D, UNet3DConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

import torch.nn as nn


class PointTransformerLightningModuleConfig(PointTransformerConfig, BaseLightningModuleConfig):
    pass


class PointTransformerLightningModule(BaseLightningModule):
    def __init__(self, module_config: PointTransformerLightningModuleConfig, ModelClass=PointTransformer):
        super().__init__(module_config, ModelClass, module_config)

        self.pos_weights = 14.1
        
    def _shared_step(self, batch, batch_idx):
        
        x, y = batch["X"], batch["Y"]
                
        y_hat = self(x)
        
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
        
        loss = torch.tensor(0.0, device=self.device)
        
        loss_layer_weights = [1.0, *self.config.loss_layer_weights[::-1]]
        
        if self.config.loss_layer_weights:
            
            y_reshaped = y
            
            # TODO: revise pos_weights
            for i in range(len(self.config.loss_layer_weights)):
                N, C, W, H, D = y_reshaped.shape
                 
                count_pos = y_reshaped.sum(dim=(1, 2, 3, 4))
                count_pos[count_pos == 0] = 1
                
                if (W * H * D < count_pos).sum():
                    raise ValueError("The number of positive voxels is greater than the total number of voxels.")
                pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss += loss_layer_weights[i] * criterion(y_hat[i], y_reshaped.float())
            
                y_reshaped = torch.nn.functional.interpolate(y_reshaped.float(), scale_factor=0.5, mode='trilinear')
                y_reshaped[y_reshaped > 0.0] = 1.0
                
            N, C, W, H, D = y_reshaped.shape
                
            count_pos = y_reshaped.sum(dim=(1, 2, 3, 4))
            count_pos[count_pos == 0] = 1
            if (W * H * D < count_pos).sum():
                    raise ValueError("The number of positive voxels is greater than the total number of voxels.")
            pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss += loss_layer_weights[-1] * criterion(y_hat[-1], y_reshaped.float())
        
        else:
            # N, C, W, H, D = y.shape
             
            # count_pos = y.sum(dim=(1, 2, 3, 4)).float().mean()
            
            # # exponentially weighted average of pos weights
            # if count_pos != 0:
            #     if self.pos_weights is None:
            #         self.pos_weights = ((W * H * D - count_pos) / count_pos)
            #     else:
            #         self.pos_weights = 0.99 * self.pos_weights + 0.01 * ((W * H * D - count_pos) / count_pos)
            # pos_weight = self.pos_weights.reshape(1, 1, 1, 1)

            # #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # criterion = nn.BCEWithLogitsLoss()
            
            N, C, W, H, D = y.shape
             
            count_pos = y.sum(dim=(1, 2, 3, 4))
            count_pos[count_pos == 0] = 1
            pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(y_hat, y.float())
            
            if self.config.num_pairs is not None:
                loss = criterion(y_hat, y.float())
            else:
                loss = criterion(y_hat, y.float())
            
        #self.log("pos_weight", self.pos_weights.item(), on_step=True)
        self.log("pos_weight", pos_weight.mean().item(), on_step=True)
            
        return loss, y_hat[0] if isinstance(y_hat, list) else y_hat, y
       
