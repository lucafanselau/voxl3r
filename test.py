from einops import rearrange
import torch
from datasets import chunk, scene, transforms
from datasets.transforms.smear_images import SmearMast3r, SmearMast3rConfig, SmearMast3rUsingVoxelizedScene
from utils.data_parsing import load_yaml_munch
from utils.transformations import extract_rot_trans, invert_pose
from visualization import Visualizer, Config


class DataConfig(chunk.occupancy.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig):
    pass


if __name__ == "__main__":
    # first load data_config
    config = DataConfig.load_from_files([
            "./config/data/base.yaml",
            "./config/data/mast3r_scenes.yaml"
        ], {
            "skip_prepare": True
        })
    config.scenes = ["8b5caf3398"]
    
    
    base_dataset = scene.Dataset(config)
    image_dataset = chunk.image.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(config, base_dataset, image_dataset),
    ], transform=None)

    zip.prepare_data()

    data = zip[0]
    
    visualizer = Visualizer(Config(log_dir=".visualization", data_dir=config.data_dir))

    #visualizer.add_from_voxel_grid(base_dataset.get_voxelized_scene(data["scene_name"]))
    
    config.mast3r_verbose = True
    smearer = SmearMast3r(config)
    smeared_chunk = smearer(data)

    # visualizer.add_points(rearrange(smeared_chunk["verbose"]["coordinates"], "C X Y Z -> (X Y Z) C"))


    # visualizer.add_from_scene_occ(data)
    visualizer.add_scene(data["scene_name"], opacity=0.1)


    # visualizer.show()   
    
    for i in range(len(zip)):
        data = zip[i]
        smeared_chunk = smearer(data)
        visualizer.add_from_occupancy_dict(smeared_chunk, opacity=0.8, transformed=True)
        visualizer.add_from_occupancy_dict_as_points(smeared_chunk, opacity=0.1, color="red")
    # visualizer.(smeared_chunk)

    # for i in range(10):
    #     data = zip[i]
    #     smeared_chunk = smearer(data)
    #     #visualizer.add_from_occupancy_dict(smeared_chunk, opacity=0.25, transformed=True)
    #     visualizer.add_from_occupancy_dict_as_points(data, opacity=0.1, color="red")
    
    # if False:
    #     visualizer.add_from_smearing_transform(smeared_chunk)
    # else:
    #     visualizer.add_from_image_dict(data)

    visualizer.add_scene(data["scene_name"], opacity=0.1)
    # visualizer.add_chunked_mesh_from_zip_dict(data, opacity=1.0)
    
    # visualizer.add_from_occupancy_dict_as_points(data, opacity=1.0, color="red")
    # visualizer.add_from_occupancy_dict(smeared_chunk, opacity=0.25, transformed=True)
    
    if False:
        visualizer.add_from_occupancy_dict_as_points(data, opacity=0.25, to_world=False)
        visualizer.add_from_occupancy_dict(data, opacity=0.25, to_world=False)
        camera_0_T_cw = data["images"][1][0]["T_cw"]
        visualizer.add_from_image_dict(data, base_coordinate_frame=camera_0_T_cw)
    
    visualizer.add_axis()

    #visualizer.show()

    visualizer.export_html("out", timestamp=True)
