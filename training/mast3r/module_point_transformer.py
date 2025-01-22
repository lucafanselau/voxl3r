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
        
    def _shared_step(self, batch, batch_idx):
    
        x, y = batch["X"], batch["Y"]
                
        y_hat, loss_mask = self(x)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat[loss_mask], y.to(y_hat)[loss_mask])
            
        return loss, y_hat, y, loss_mask
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y, loss_mask = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        #self.train_metrics(probs[loss_mask], y.int()[loss_mask])
        self.train_metrics(probs, y.int())

        # Log everything
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, loss_mask = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        #self.train_metrics(probs[loss_mask], y.int()[loss_mask])
        self.val_metrics(probs, y.int())

        # Log everything
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y, loss_mask = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        #self.train_metrics(probs[loss_mask], y.int()[loss_mask])
        self.test_metrics(probs, y.int())
        
        # Log everything
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)

        self.test_precision_recall.update(probs, y.int())

        # y_true = y.int().flatten()
        # y_pred = probs.flatten().unsqueeze(1)

        # # y_pred needs to be a mapping (N, 2) where the first is the probability of the zero class and the second is the probability of the one class
        # y_pred = torch.cat((1 - y_pred, y_pred), dim=1)

        # curve = wandb.plot.pr_curve(y_true=y_true.cpu(), y_probas=y_pred.cpu(), labels=["0", "1"])
        # self.logger.experiment.log({"pr_curve": curve})

        return {"loss": loss, "pred": y_hat.detach().cpu()} 
       
