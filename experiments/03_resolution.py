from training.mast3r import train
from training.tuner import tune

from scipy.stats.distributions import loguniform

def main():
    # first load data_config
    data_config = train.DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
    ])
    config = train.Config.load_from_files([
        "./config/trainer/tune.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml"
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels()
    })
    config.skip_prepare = True
    config.skip_connections = False

    search_space = {
        # direction: first index is lowest res loss
        "grid_resolution": [0.02, 0.04],
        #"skip_dropout_p": [0.75, 1]
    }

    experiment_name = "02_proxy"
    config.name = "mast3r-3d-experiments"

    tune(train.train, config, search_space, num_samples=2, experiment_name=experiment_name)

if __name__ == "__main__":
    main()
