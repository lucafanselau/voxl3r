from training.mast3r import train_transformer
from training.tuner import tune

from scipy.stats.distributions import loguniform

def main():
    # first load data_config
    data_config = train_transformer.DataConfig.load_from_files([
    "./config/data/base.yaml",
    "./config/data/undistorted_scenes.yaml"
    ])
    data_config.add_confidences = True
    data_config.add_pts3d = True
    config = train_transformer.Config.load_from_files([
        "./config/trainer/tune.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/network/transformer.yaml",
        "./config/module/base.yaml"
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels()
    })
    config.skip_connections = False
    config.with_downsampling = True
    config.with_learned_pooling = True
    config.refinement_layers = 0
    
    config.use_initial_batch_norm = True
        
    config.skip_prepare = True
    config.base_channels = 64
    config.use_learned_pe = True
    
    config.dim = 256
    config.dim_head = 64
    config.mlp_dim = 512
    config.channels = 256
    
    #config.weight_decay = 0.0005
    
    search_space = {
        # direction: first index is lowest res loss
        "refinement_layers": [0],
        #"skip_dropout_p": [0.75, 1]
    }

    experiment_name = "08_transformer_no_refinement"
    config.name = "mast3r-3d-experiments"

    tune(train_transformer.train, config, search_space, num_samples=1, experiment_name=experiment_name, base_epochs=30, final_epochs=0)

if __name__ == "__main__":
    main()
