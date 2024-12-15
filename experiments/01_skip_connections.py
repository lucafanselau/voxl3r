from training.mast3r import train
from training.tuner import tune

from scipy.stats.distributions import loguniform

def main():
    # first load data_config
    data_config = train.DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/mast3r_scenes.yaml"
    ])
    config = train.Config.load_from_files([
        "./config/trainer/tune.yaml",
        "./config/network/base_unet.yaml",
        "./config/module/base.yaml"
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels()
    })
    config.skip_prepare = True
    config.max_epochs = 2

    search_space = {
        "skip_connections": [True, False]
    }

    experiment_name = "01_sc"

    tune(train.train, config, search_space, num_samples=8, experiment_name=experiment_name)

if __name__ == "__main__":
    main()