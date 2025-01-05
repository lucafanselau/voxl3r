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
    config.weight_decay = 0.0005
    
    search_space = {
        # direction: first index is lowest res loss
        "refinement_blocks": ["inceptionBlockA"],
        #"skip_dropout_p": [0.75, 1]
    }

    experiment_name = "07_weight_decay"
    config.name = "mast3r-3d-experiments"

    tune(train.train, config, search_space, num_samples=1, experiment_name=experiment_name, base_epochs=40, final_epochs=0)

if __name__ == "__main__":
    main()
