from einops import rearrange
import torch
from datasets import chunk, scene, transforms
from datasets.transforms.smear_images import SmearMast3r, SmearMast3rConfig, SmearMast3rUsingVoxelizedScene
from training.default.data import DefaultDataModule
from utils.data_parsing import load_yaml_munch
from utils.transformations import extract_rot_trans, invert_pose
from visualization import Visualizer, Config


class DataConfig(chunk.occupancy_revised.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig):
    pass


def main():
    # first load data_config
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
    ])


    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml",
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
    })
    
    config.mast3r_verbose = True
    
    scene_name = data_config.scenes[13]
    data_config.scenes = [scene_name]
    config.scenes = data_config.scenes
    
    # Train
    base_dataset = scene.Dataset(data_config)
    base_dataset.prepare_data()
    image_dataset = chunk.image.Dataset(data_config, base_dataset)

    # zip = chunk.zip.ZipChunkDataset([
    #     image_dataset,
    #     #chunk.occupancy.Dataset(data_config, base_dataset, image_dataset),
    #     chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    # ], transform=transforms.SmearMast3rUsingVoxelizedScene(data_config), base_dataset=base_dataset)
    
    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(data_config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    ], transform=SmearMast3r(config))
    
    zip.prepare_data()
    
    visualizer = Visualizer(Config(log_dir=".visualization", data_dir=config.data_dir))

    visualizer.add_scene(config.scenes[0], opacity=0.1)
    
    # smearer = SmearMast3r(config)
    # smeared_chunk = smearer(data)

    # # visualizer.add_points(rearrange(smeared_chunk["verbose"]["coordinates"], "C X Y Z -> (X Y Z) C"))


    # # visualizer.add_from_scene_occ(data)
    # visualizer.add_scene(data["scene_name"], opacity=0.1)


    # # visualizer.show()   
    
    data = zip[11]
    visualizer.add_from_occupancy_dict(data, opacity=0.1, transformed=True)
    #visualizer.add_from_occupancy_dict_as_points(data, opacity=0.1, color="red", with_transform=True)
    visualizer.add_from_image_dict(data["verbose"]["data_dict"])

    # # for i in range(10):
    # #     data = zip[i]
    # #     smeared_chunk = smearer(data)
    # #     #visualizer.add_from_occupancy_dict(smeared_chunk, opacity=0.25, transformed=True)
    # #     visualizer.add_from_occupancy_dict_as_points(data, opacity=0.1, color="red")
    
    # # if False:
    # #     visualizer.add_from_smearing_transform(smeared_chunk)
    # # else:
    # #     visualizer.add_from_image_dict(data)

    # visualizer.add_scene(data["scene_name"], opacity=0.1)
    # # visualizer.add_chunked_mesh_from_zip_dict(data, opacity=1.0)
    
    # # visualizer.add_from_occupancy_dict_as_points(data, opacity=1.0, color="red")
    # # visualizer.add_from_occupancy_dict(smeared_chunk, opacity=0.25, transformed=True)
    
    # if False:
    #     visualizer.add_from_occupancy_dict_as_points(data, opacity=0.25, to_world=False)
    #     visualizer.add_from_occupancy_dict(data, opacity=0.25, to_world=False)
    #     camera_0_T_cw = data["images"][1][0]["T_cw"]
    #     visualizer.add_from_image_dict(data, base_coordinate_frame=camera_0_T_cw)
    
    visualizer.add_axis()

    # #visualizer.show()

    visualizer.export_html("out", timestamp=True)


    


if __name__ == "__main__":
    main()