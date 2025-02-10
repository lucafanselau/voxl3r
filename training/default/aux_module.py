from typing import Optional, Tuple, Union, Type
from pydantic import Field
import torch.nn as nn

from datasets import chunk, transforms, transforms_batched
from datasets.transforms.sample_coordinate_grid import SampleCoordinateGridConfig
from networks.surfacenet import SurfaceNet
from networks.u_net import UNet3DConfig
from training.common import load_config_from_checkpoint
from training.default.data import DefaultDataModuleConfig
from training.default.module import BaseLightningModule, BaseLightningModuleConfig
from utils.config import BaseConfig

class LightningModuleWithAuxConfig(BaseLightningModuleConfig):
    project_name: str = "mast3r-3d-experiments"
    run_name: str = "adkef9ex"
    checkpoint_name: str = "last"
    config_class: str = "TrainConfig"
    model_class: str = "SurfaceNet"
    train_aux: bool = False
    
    
class DataConfig(chunk.mast3r.Config, chunk.occupancy_revised.Config, chunk.image_loader_compressed.Config, transforms.SmearMast3rConfig, DefaultDataModuleConfig, transforms.ComposeTransformConfig, transforms_batched.ComposeTransformConfig, transforms_batched.SampleOccGridConfig, SampleCoordinateGridConfig):
    name: str = "mast3r-3d"

class LoggingConfig(BaseConfig):
    grid_occ_interval: Tuple[int, int, int] = Field(default=(3, 3, 1))
    save_top_k: int = 1
    log_every_n_steps: int = 1

class TrainerConfig(BaseConfig):
    max_epochs: int = 300
    # overrides only the max_epochs for the trainer, not the max_epochs for the lr_scheduler and tuner
    limit_epochs: Optional[int] = None
    limit_val_batches: int = 16
    check_val_every_n_epoch: int = 1


class TrainConfig(LoggingConfig, UNet3DConfig, BaseLightningModuleConfig, TrainerConfig, DataConfig):
    resume: Union[bool, str] = False
    checkpoint_name: str = "last"
    
configs = {
    "TrainConfig": TrainConfig,
}

models = {
    "SurfaceNet": SurfaceNet,
}

class LightningModuleWithAux(BaseLightningModule):
    def __init__(
        self,
        module_config: LightningModuleWithAuxConfig,
        ModelClass: Union[Type[nn.Module], list],
    ):
        super().__init__(module_config, ModelClass)
        
        config, path = load_config_from_checkpoint(module_config.project_name, module_config.run_name, ConfigClass=configs[module_config.config_class], checkpoint_name=module_config.checkpoint_name)
        aux_model = BaseLightningModule.load_from_checkpoint(path, module_config=config, ModelClass=models[module_config.model_class]).model
        
        if aux_model is not None:
            if not isinstance(self.model, nn.ModuleList):
                self.model = nn.ModuleList([self.model])
                
            if not module_config.train_aux:
                aux_model.requires_grad_(False)
            self.model.insert(0, aux_model)
