import torch
from networks.u_net import UNet3D, UNet3DConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

import torch.nn as nn


class Unet3DLightningModuleConfig(UNet3DConfig, BaseLightningModuleConfig):
    pass


class UNet3DLightningModule(BaseLightningModule):
    def __init__(self, module_config: Unet3DLightningModuleConfig):
        super().__init__(module_config, UNet3D, module_config)
        
    def _shared_step(self, batch, batch_idx):
        
        x, y = batch["X"], batch["Y"]
        
        y_hat = self(x)


        
        loss = torch.tensor(0.0, device=self.device)
        
        loss_layer_weights = [1.0, *self.config.loss_layer_weights[::-1]]
        
        if self.config.loss_layer_weights:
            
            y_reshaped = y
            
            for i in range(len(self.config.loss_layer_weights)):
                N, C, W, H, D = y_reshaped.shape
                 
                count_pos = y_reshaped.sum(dim=(1, 2, 3, 4))
                count_pos[count_pos == 0] = 1
                pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss += loss_layer_weights[i] * criterion(y_hat[i], y_reshaped.float())
            
                y_reshaped = torch.nn.functional.interpolate(y_reshaped.float(), scale_factor=0.5, mode='trilinear')
                y_reshaped[y_reshaped > 0.0] = 1.0
                
            N, C, W, H, D = y_reshaped.shape
                
            count_pos = y_reshaped.sum(dim=(1, 2, 3, 4))
            count_pos[count_pos == 0] = 1
            pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss += loss_layer_weights[-1] * criterion(y_hat[-1], y_reshaped.float())
        
        else:
            N, C, W, H, D = y.shape
             
            count_pos = y.sum(dim=(1, 2, 3, 4))
            count_pos[count_pos == 0] = 1
            pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(y_hat, y.float())
            
        return loss, y_hat[0], y
       
