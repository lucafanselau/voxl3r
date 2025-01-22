from argparse import ArgumentParser
import os
from pathlib import Path
import traceback
from typing import Optional, Tuple, Union
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from datasets.transforms.point_based import PointBasedTransform
from datasets.transforms.smear_images import SmearMast3r
from networks.u_net import UNet3DConfig
from pydantic import Field
from loguru import logger

import torch
from datasets import chunk, transforms, scene
from networks.volume_transformer import VolumeTransformerConfig
from networks.voxel_based import VoxelBasedNetworkConfig
from training.common import create_datasets
from training.mast3r.module_transformer_unet3D import TransformerUNet3DLightningModule
from training.mast3r.module_unet3d import UNet3DLightningModule
from training.mast3r.module_voxel_based import VoxelBasedLightningModule
from utils.config import BaseConfig

from training.loggers.occ_grid import OccGridCallback
from training.default.data import DefaultDataModuleConfig, DefaultDataModule
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

class DataConfig(chunk.occupancy_revised.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig, DefaultDataModuleConfig):
    name: str = "mast3r-3d"

class LoggingConfig(BaseConfig):
    grid_occ_interval: Tuple[int, int, int] = Field(default=(4, 4, 1))
    save_top_k: int = 3
    log_every_n_steps: int = 1

class TrainerConfig(BaseConfig):
    max_epochs: int = 300
    # overrides only the max_epochs for the trainer, not the max_epochs for the lr_scheduler and tuner
    limit_epochs: Optional[int] = None
    limit_val_batches: int = 16
    check_val_every_n_epoch: int = 1


class Config(LoggingConfig, VolumeTransformerConfig, UNet3DConfig, BaseLightningModuleConfig, TrainerConfig, DataConfig):
    resume: Union[bool, str] = False
    checkpoint_name: str = "last"

# class Config(LoggingConfig, VoxelBasedNetworkConfig, BaseLightningModuleConfig, TrainerConfig, DataConfig):
#     resume: Union[bool, str] = False
#     checkpoint_name: str = "last"

def train(
        config: dict, 
        default_config: Config, 
        trainer_kwargs: dict = {}, 
        identifier: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):

    config: Config = Config(**{**default_config.model_dump(), **config}) 

    logger.debug(f"Config: {config}")


    RESUME_TRAINING = config.resume != False
    data_config = config

    torch.set_float32_matmul_precision("medium")

    # if we have an identifier, we also want to check if we have a wandb run with that identifier
    if identifier is not None:
        last_ckpt_folder = Path(f"./.lightning/{data_config.name}/{data_config.name}/{identifier}")
        if last_ckpt_folder.exists():
            logger.info(f"Found wandb run {identifier}, resuming training")
            RESUME_TRAINING = True
        else:
            last_ckpt_folder = None
    elif RESUME_TRAINING:
        if not isinstance(config.resume, str):
            ckpt_folder = list(Path(f"./.lightning/{data_config.name}/{data_config.name}/").glob("*"))
            ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
            last_ckpt_folder = ckpt_folder[-1]
        else:
            last_ckpt_folder = Path(f"./.lightning/{data_config.name}/{data_config.name}/{config.resume}")
    else:
        last_ckpt_folder = None

    # Setup logging
    if RESUME_TRAINING:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
            id=last_ckpt_folder.stem,
            name=run_name,
            resume="allow",
            group=experiment_name,
        )
    else:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
            id=identifier,
            name=run_name,
            group=experiment_name,
        )
        

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Checkpointing callbacks
    filename = "{epoch}-{step}"
    callbacks = [
        ModelCheckpoint(
            filename=filename + f"-{{{type}_{name}:.2f}}",
            monitor=f"{type}_{name}",
            save_top_k=config.save_top_k,
            mode=mode,
        )
        for type in ["train", "val"]
        for [name, mode] in [
            ["loss", "min"],
            ["accuracy", "max"],
            ["precision", "max"],
            ["recall", "max"],
            #["f1", "max"],
            # ["auroc", "max"],
        ]
    ]
    # Save the model every 5 epochs
    last_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
    )

    # Custom callback for logging the 3D voxel grids
    voxel_grid_logger = OccGridCallback(wandb=wandb_logger, n_epochs=config.grid_occ_interval)

    # 

    datamodule = create_datasets(config, splits=["train", "val"], transform=transforms.SmearMast3r)   
    datamodule.prepare_data()
    # Create configs
    #device_stats = DeviceStatsMonitor(cpu_stats=True)

    # module = VoxelBasedLightningModule(module_config=config) 
    module = TransformerUNet3DLightningModule(module_config=config)
    
    wandb_logger.watch(module.model, log=None, log_graph=True)

    # Initialize trainer
    trainer_args = {
        **trainer_kwargs,
        "max_epochs": config.max_epochs if config.limit_epochs is None else config.limit_epochs,
        # "profiler": "simple",
        "log_every_n_steps": config.log_every_n_steps,
        #"callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, voxel_grid_logger, lr_monitor, device_stats],
        "callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, lr_monitor],
        "logger": wandb_logger,
        "precision": "bf16-mixed", 
        "default_root_dir": "./.lightning/mast3r-3d",
        "limit_val_batches": config.limit_val_batches,
        # overfit settings
        # "overfit_batches": 1,
        # "check_val_every_n_epoch": None,
        # "val_check_interval": 4000,
    }
    
    #profiler = AdvancedProfiler(dirpath="./profiler_logs", filename="perf_logs")
    trainer = Trainer(
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        **trainer_args,
        #profiler=profiler
    )

    if RESUME_TRAINING:
        logger.info(f"Resuming training from {last_ckpt_folder}")

    try:
        trainer.fit(
            module,
            datamodule=datamodule,
            ckpt_path=(
                last_ckpt_folder / f"checkpoints/{config.checkpoint_name}.ckpt" if RESUME_TRAINING else None
            ),
        )
    except Exception as e:
        logger.error(f"Training failed with error, finishing training")
        print(traceback.format_exc())
    finally:
        return trainer, module
    # finally:
        # Save best checkpoints info
        # base_path = Path(callbacks[0].best_model_path).parents[1]
        # result = base_path / "best_ckpts.pt"
        # result_dict = {
        #     f"best_model_{type}_{name}": callbacks[i * 4 + j].best_model_path
        #     for i, type in enumerate(["train", "val"])
        #     for j, [name, mode] in enumerate([
        #         ["loss", "min"],
        #         ["accuracy", "max"],
        #         ["f1", "max"],
        #         ["auroc", "max"],
        #     ])
        # }
        # result_dict["last_model_path"] = every_five_epochs.best_model_path
        # torch.save(result_dict, result)
        # return result_dict, module, trainer, datamodule


def main():
    # first load data_config
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
        #"./config/data/undistorted_scenes.yaml"
    ])
    
    parser = ArgumentParser()

    parser.add_argument("--resume", action="store_true", help="Resume training from *last* checkpoint")
    parser.add_argument("--resume-run", dest="resume_run", type=str, help="Resume training from a specific run")
    parser.add_argument("--ckpt", dest="checkpoint_name", type=str, default="last", help="Name of the checkpoint to resume from")
    args = parser.parse_args()
    logger.info(f"Parsed args: {args}")

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/network/transformer.yaml",
        "./config/module/base.yaml",
    ], {
        **vars(args),
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
        "resume": args.resume_run if args.resume_run is not None else args.resume,
    })
 
    config.max_epochs = 30
    config.skip_connections = False
    config.learning_rate = 0.0005
    config.weight_decay = 0.00001
    config.mlp_dim = 512
    
    config.refinement_blocks = "inceptionBlockA"
    config.name = "mast3r-3d-experiments"
    config.num_pairs = 4
    config.use_initial_batch_norm = True

    config.skip_prepare = True
    # config.with_downsampling = False

    import numpy as np

    #"batch_size=12 num_workers=11 val_num_workers=5 pe_enabled=True add_projected_depth=False add_validity_indicator=False add_viewing_directing=False grid_sampling_mode='bilinear' concatinate_pe=False shuffle_images=True seq_len=8 add_confidences=True add_pts3d=True ouput_only_training=False mast3r_verbose=False seperate_image_pairs=True data_dir='/mnt/data/scannetpp/data' camera='dslr' scenes=['02455b3d20', '1841a0b525', '2e67a32314', '45d2e33be1', '5ee7c22ba0', '7bc286c1b6', '954633ea01', 'ab6983ae6c', 'bfd3fd54d2', 'd4d2019f5d', 'ebff4de90b', '02a980c994', '18fd041970', '2e74812d00', '471cc4ba84', '5f99900f09', '7cd2ac43b4', '95d525fbfd', 'ac48a9b736', 'c06a983e63', 'd6419f6478', 'ecb5d01065', '1a130d092a', '302a7f6b67', '47b37eb6f9', '5fb5d2dbf2', '7d72f01865', '961911d451', 'accad58571', 'c0c863b72d', 'd662592b54', 'ed2216380b', '03f7a0e617', '1a8e0d78c0', '303745abc7', '47eb87b5bb', '6115eddb86', '7dfdff1b7d', '9663292843', 'acd69a1746', 'c0cbb1fea1', 'd6702c681d', 'eeddfe67f5', '047fb766c4', '1ada7a0617', '30966f4c6e', '480ddaadc0', '61adeff7d5', '7e09430da7', '9859de300f', 'acd95847c5', 'c0f5742640', 'd6cbe4b28b', 'ef18cf0708', '0529d56cce', '1ae9e5d2a6', '30f4a2b44d', '484ad681df', '633f9a9f06', '7e7cd69a59', '98b4ec142f', 'ad2d07fd11', 'c173f62b15', 'd6d9ddb03f', 'ef25276c25', '06a3d79b68', '1b75758486', '319787e6ec', '497588b572', '6464461276', '7eac902fd5', '98fe276aa8', 'ada5304e41', 'c24f94007b', 'd755b3d9d8', 'ef69d58016', '076c822ecc', '1b9692f0c7', '31a2c91c43', '49a82360aa', '646af5e14b', '7f4d173c9c', '99fa5c25e1', 'b074ca565a', 'c3e279be54', 'd7abfc4b17', 'f00bd5fa8a', '079a326597', '1be2c31cac', '320c3af000', '49c758655e', '64ea6b73c2', '7ffc86edf4', '9a9e32c768', 'b08a908f0f', 'c413b34238', 'd918af9c5f', 'f07340dfea', '07f5b601ee', '1c4b893630', '32280ecbca', '4a1a3a7dc5', '651dc6b4f1', '80ffca8a48', '9b365a9b68', 'b09431c547', 'c47168fab2', 'da8043d54e', 'f0b0a42ba3', '07ff1c45bb', '1c876c250f', '324d07a5b3', '4ba22fa7e4', '66c98f4a9b', '8133208cb6', '9b74afd2d2', 'b09b119466', 'c49a8c6cff', 'daffc70503', 'f20e7b5640', '08bbbdcc3d', '1d003b07bd', '355e5e32db', '4bc04e0cde', '67d702f2e8', '824d9cfa6e', '9e019d8be1', 'b0a08200c9', 'c4c04e6d6c', 'dc263dfbf0', 'f248c2bcdc', '09bced689e', '1f7cbbdde1', '37ea1c52f0', '4c5c60fa76', '6855e1ac32', '825d228aec', '9f139a318d', 'b1d75ecd55', 'c50d2d1d42', 'dfac5b38df', 'f25f5e6f63', '09c1414f1b', '210f741378', '3864514494', '4ea827f5a1', '68739bdf1f', '8283161f1b', '9f21bdec45', 'b20a261fdf', 'c5439f4607', 'dfe9cbd72a', 'f2dc06b1d2', '0a184cf634', '21532e059d', '38d58a7a31', '4ef75031e3', '689fec23d7', '84b48f2614', '9f7641ce94', 'b26e64c4b0', 'c545851c4f', 'dffce1cf9a', 'f34d532901', '0a5c013435', '21d970d8de', '3928249b53', '50809ea0d8', '69e5939669', '85251de7d1', '9f79564dbf', 'b5918e4637', 'c5f701a8c7', 'e01b287af5', 'f3685d06a9', '0a76e06478', '251443268c', '394a542a19', '52599ae063', '6b40d1a939', '87f6d7d564', 'a003a6585e', 'b73f5cdc41', 'c856c41c99', 'e050c15a8d', 'f3d64c30f8', '0a7cc12c0e', '252652d5ba', '39e6ee46df', '5298ec174f', '6cc2231b9c', '88627b561e', 'a05ee63164', 'b97261909e', 'c8f2218ee2', 'e0abd740ba', 'f5401524e5', '0b031f3119', '25927bb04c', '39f36da05b', '5371eff4f9', '6d89a7320d', '8890d0a267', 'a08d9a2476', 'ba414a3e6f', 'c9abde4c4b', 'e0de253456', 'f6659a3107', '0cf2e9402d', '25f3b7a318', '3a161a857d', '54b6127146', '6ebe30292e', '88cf747085', 'a08dda47a8', 'bb87c292ad', 'c9bf4c8b62', 'e0e83b4ca3', 'f7a60ba2a2', '0d2ee665be', '260db9cf5a', '3c95c89d61', '54bca9597e', '6ee2fc1070', '89214f3ca0', 'a1d9da703c', 'bc03d88fc3', 'ca0e09014e', 'e1b1d9de55', 'f8062cb7ce', '0e75f3c4d9', '260fa55d50', '3db0a1c8f3', '54e7ffaea3', '6f12492455', '893fb90e89', 'a24858e51e', 'bc2fce1d81', 'cbd4b3055e', 'e398684d27', 'f8f12e4e6b', '104acbf7d2', '27dd4da69e', '3e6ceea56c', '55b2bf8036', '6f1848d1e3', '8a20d62ac0', 'a24f64f7fb', 'bc400d86e1', 'cc49215f67', 'e3ecd49e2b', 'f94c225e84', '108ec0b806', '280b83fcf3', '3e8bba0176', '5654092cc2', '7079b59642', '8a35ef3cfe', 'a29cccc784', 'bcd2436daf', 'cc5237fd77', 'e6afbe3753', 'f9f95681fd', '116456116b', '281ba69af1', '3e928dc2f6', '5656608266', '709ab5bffe', '8b2c0938d6', 'a46b21d949', 'bd7375297e', 'ccdc33dc2a', 'e7ac609391', 'faec2f0468', '11b696efba', '281bc17764', '3f15a9266d', '569f99f881', '712dc47104', '8b5caf3398', 'a4e227f506', 'bd9305480d', 'ccfd3ed9c7', 'e7af285f7d', 'fb05e13ad1', '1204e08f17', '285efbc7cf', '3f1e1610de', '56a0ec536c', '728daff2a3', '8be0cd3817', 'a5114ca13d', 'bde1e479ad', 'cd2994fcc1', 'e898c76c1f', 'fb5a96b1a2', '124974734e', '286b55a2bf', '40aec5fffa', '5748ce6f01', '74ff105c0d', '8d563fc2cc', 'a897272241', 'be0ed6b33c', 'cd7973d92b', 'e8e81396b6', 'fd361ab85f', '13285009a4', '28a9ee4557', '40b56bf310', '578511c8a9', '75d29d69b8', '8e00ac7f59', 'a8bf42d646', 'be2e10f16a', 'cd88899edb', 'e8ea9b4da8', 'fe1733741f', '1366d5ae89', '290ef3f2c9', '410c470782', '5942004064', '77596f5d2a', '8e6ff28354', 'a980334473', 'be6205d016', 'cf1ffd871d', 'e91722b5a3', 'fe94fc30cf', '13c3e046d7', '2970e95b65', '419cbe7c11', '59e3f1ea37', '7831862f02', '8f82c394d6', 'aa6e508f0c', 'be66c57b92', 'd070e22e3b', 'e9ac2fc517', '15155a88fb', '29b607c6d5', '41b00feddb', '5a14f9da39', '785e7504b9', '9071e139d9', 'aaa11940d3', 'be91f7884d', 'd1b9dff904', 'e9e16b6043', '154c3e10d9', '2a1a3afad9', '4318f8bb3c', '5a269ba6fe', '7977624358', '9460c8889d', 'ab046f8faf', 'beb802368c', 'd228e2d9dd', 'ea15f3457c', '16c9bd2e1e', '2a496183e1', '4422722c49', '5d152fab1b', '7b37cccb03', '9471b8d485', 'ab11145646', 'bee11d6a41', 'd2f44bf242', 'eb4bc76767', '1831b3823a', '2b1dc6d6a5', '45b0dac5e3', '5eb31827b7', '7b6477cb95', '94ee15e8ba', 'ab252b28c0', 'bf6e439e38', 'd415cc449b', 'ebc200e928'] storage_preprocessing_voxelized_scenes='preprocessed_voxel_grids' num_workers_voxelization=2 force_prepare_voxelize=False scene_resolution=0.01 return_voxelized=True storage_preprocessing='preprocessed' chunk_num_workers=11 force_prepare=False overfit_mode=False skip_prepare=True with_furthest_displacement=True center_point=array([0.  , 0.  , 1.28]) folder_name_image='corresponding_images' mast3r_data_dir='/mnt/dorta/mast3r/data' folder_name_mast3r='mast3r_preprocessed' batch_size_mast3r=8 force_prepare_mast3r=False grid_resolution=0.04 grid_size=[32, 32, 32] folder_name_occ='prepared_occ_grids' force_prepare_occ=False batch_size_occ=8 name='mast3r-3d-experiments' max_epochs=30 limit_epochs=None limit_val_batches=32 check_val_every_n_epoch=2 learning_rate=0.0005 scheduler_factor=0.5 scheduler_patience=5 weight_decay=1e-05 eta_min=5e-05 scheduler='CosineAnnealingLR' in_channels=56 base_channels=32 num_layers=2 refinement_layers=2 refinement_bottleneck=2 skip_connections=False disable_norm=False with_downsampling=False with_learned_pooling=False keep_dim_during_up_conv=False refinement_blocks='inceptionBlockA' use_initial_batch_norm=True skip_dropout_p=0.0 loss_layer_weights=[] num_pairs=4 cube_size=8 cube_patch_size=1 dim=128 depth=8 heads=4 mlp_dim=512 channels=128 dim_head=32 no_attn_feedthrough=False use_learned_pe=False grid_occ_interval=(4, 4, 1) save_top_k=3 log_every_n_steps=1 resume=False checkpoint_name='last'" 
    config.batch_size = 12
    config.num_workers = 11
    config.val_num_workers = 5
    config.pe_enabled = True
    config.add_projected_depth = False
    config.add_validity_indicator = False
    config.add_viewing_directing = False
    config.grid_sampling_mode = 'bilinear'
    config.concatinate_pe = False
    config.shuffle_images = False
    config.seq_len = 5
    config.seperate_image_pairs = True
    config.num_workers_voxelization = 2
    config.force_prepare_voxelize = False
    config.scene_resolution = 0.01
    config.return_voxelized = True
    config.chunk_num_workers = 11
    config.force_prepare = False
    config.overfit_mode = False
    config.with_furthest_displacement = True
    config.center_point = np.asarray([0.0, 0.0, 1.5])
    config.batch_size_mast3r = 8
    config.force_prepare_mast3r = False
    config.grid_resolution = 0.04
    config.grid_size = [32, 32, 32]
    config.force_prepare_occ = False
    config.batch_size_occ = 8
    config.limit_val_batches = 32
    config.check_val_every_n_epoch = 2
    config.scheduler_factor = 0.5
    config.scheduler_patience = 5
    config.eta_min = 5e-05
    config.scheduler = 'CosineAnnealingLR'
    config.base_channels = 32
    config.num_layers = 2
    config.num_refinement_blocks = 2
    config.refinement_bottleneck = 2
    config.disable_norm = False
    config.keep_dim_during_up_conv = False
    config.skip_dropout_p = 0.0
    config.loss_layer_weights = []
    config.cube_size = 8
    config.cube_patch_size = 1
    config.dim = 128
    config.depth = 8
    config.heads = 4
    config.channels = 128
    config.dim_head = 32
    config.no_attn_feedthrough = False
    config.grid_occ_interval = (4, 4, 1)
    config.save_top_k = 3
    config.log_every_n_steps = 1

    train({}, config, experiment_name="08_trial_transformer_unet3d")


if __name__ == "__main__":
    main()



    

