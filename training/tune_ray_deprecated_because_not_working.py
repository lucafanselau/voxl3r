from lightning import Trainer
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayFSDPStrategy,
    RayDeepSpeedStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.search.bayesopt import BayesOptSearch
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

from training.mast3r.train import Config
from networks.u_net import Simple3DUNetConfig
from training.default.module import BaseLightningModuleConfig
from training.mast3r.train import DataConfig, TrainerConfig, train
from utils.config import BaseConfig
import os


class BaseTuneConfig(BaseConfig):
    max_epochs: int
    init_tune_steps: int
    total_tune_steps: int
    



def custom_tune(train_func, default_config: Config, tune_settings: BaseTuneConfig, search_space: dict):


    ray.init(local_mode=True, num_cpus=8, num_gpus=1)

    trainer_kwargs = {
        "devices": "auto",
        "accelerator": "auto",
        # "strategy": RayFSDPStrategy(),
        "strategy": RayDDPStrategy(find_unused_parameters=True),
        # "strategy": RayDeepSpeedStrategy(),
        "callbacks": [RayTrainReportCallback()],
        "plugins": [RayLightningEnvironment()],
    }

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="train_loss",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        tune.with_parameters(train_func, default_config=default_config, trainer_kwargs=trainer_kwargs),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    scheduler = ASHAScheduler(
        max_t=tune_settings.max_epochs, grace_period=1, reduction_factor=2
    )


    #algo = BayesOptSearch(random_search_steps=tune_settings.init_tune_steps)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="train_loss",
            mode="min",
            num_samples=tune_settings.total_tune_steps,
            scheduler=scheduler,
            search_alg=None,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":

    

    # first load data_config
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/mast3r_scenes.yaml"
    ])

    train_config = TrainerConfig.load_from_files([
        "./config/trainer/base.yaml"
    ])

    network_config = Simple3DUNetConfig.load_from_files([
        "./config/model/base_unet.yaml"
    ], default={ "in_channels": data_config.get_feature_channels() })

    module_config = BaseLightningModuleConfig.load_from_files([
        "./config/module/base.yaml"
    ])

    config = Config(
        data_config=data_config,
        network_config=network_config,
        module_config=module_config,
        trainer_config=train_config,
    )
    
    tune_config = BaseTuneConfig.load_from_files(["./config/tune/base.yaml"])

    search_space = {
    "module_config" :
        {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
        }
    }
    custom_tune(train, config, tune_config, search_space)
