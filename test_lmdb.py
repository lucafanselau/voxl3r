# first load data_config
from training.mast3r.train_rgb import Config as TrainConfig
from training.common import DataConfigRGBLMDB, create_dataset_rgb, create_datamodule_rgb, create_dataset_rgb_lmdb, create_datamodule_rgb_lmdb

data_config = DataConfigRGBLMDB.load_from_files([
    "./config/data/base.yaml",
    "./config/data/images.yaml"
], {
    "access": "get"
})

class Config(TrainConfig,DataConfigRGBLMDB):
    name: str = "mast3r-3d-experiments"
    pass

config = Config.load_from_files([
    "./config/trainer/base.yaml",
    "./config/network/base_unet.yaml",
    "./config/network/unet3D.yaml",
    "./config/module/base.yaml",
], {
    **data_config.model_dump(),
    "in_channels": data_config.get_feature_channels(),
})

# config.scenes = ["fd361ab85f",
#                 "d7abfc4b17",  
#                 "d918af9c5f",
#                 "da8043d54e",
#                 "dfac5b38df",
#                 "dfe9cbd72a",
#                 "dffce1cf9a",
#                 "e01b287af5",
#                 "e050c15a8d",
#                 "e0abd740ba",
#                 "e0de253456",
#                 "e0e83b4ca3",
#                 "e1b1d9de55"]

config.base_channels = 16
config.name = "mast3r-3d-experiments"
config.max_epochs = 1
config.prefetch_factor = 2
config.num_workers = 0
config.val_num_workers = 6
config.skip_prepare = True


# dataset = create_dataset_rgb_lmdb(config, split="train")
datamodule = create_datamodule_rgb_lmdb(config, splits=["train"])

datamodule.prepare_data()

dataloader = datamodule.train_dataloader()

from tqdm import tqdm

# time what a whole epoch looks like
import time
start = time.time()
for i, batch in enumerate(tqdm(dataloader)):
    pass
end = time.time()
print(f"Time for batches: {end - start}")