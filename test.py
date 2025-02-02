from multiprocessing import Manager
from einops import rearrange
import torch
from datasets import chunk, scene, transforms, transforms_batched
from datasets.transforms.smear_images import SmearMast3r
from training.default.data import DefaultDataModuleConfig
from visualization import Visualizer, Config


class DataConfig(chunk.occupancy_revised.Config, chunk.image_loader.Config, transforms.SmearImagesConfig, DefaultDataModuleConfig, transforms.ComposeTransformConfig, transforms_batched.ComposeTransformConfig, transforms_batched.SampleOccGridConfig):
    pass


def main():
    # first load data_config
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/images_transform.yaml",
        "./config/data/images_transform_batched.yaml",
    ])
    
    data_config.smear_images_verbose = True

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml",
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
    })
    
    base_dataset = scene.Dataset(data_config)
    base_dataset.prepare_data()    
    
    #scene_name = base_dataset.scenes[-1] #data_config.scenes[42]
    scene_name = base_dataset.scenes[42]
    print(f"Currently used scene is {scene_name}")
    data_config.scenes = [scene_name]
    config.scenes = data_config.scenes
    
    config.mast3r_verbose = True
    
    # Train
    base_dataset = scene.Dataset(data_config)
    base_dataset.prepare_data()    
    image_dataset = chunk.image_loader.Dataset(data_config, base_dataset)
    image_dataset.prepare_data()
    
    

    # zip = chunk.zip.ZipChunkDataset([
    #     image_dataset,
    #     #chunk.occupancy.Dataset(data_config, base_dataset, image_dataset),
    #     chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    # ], transform=transforms.SmearMast3rUsingVoxelizedScene(data_config), base_dataset=base_dataset)
    transform = transforms.ComposeTransforms(data_config)
    manager = Manager()
    transform.transforms = [transform(data_config, manager.dict()) for transform in transform.transforms]
    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(data_config, base_dataset, image_dataset),
    ], transform=transform)
    
    zip.prepare_data()
    
    visualizer = Visualizer(Config(log_dir=".visualization", **data_config.model_dump()))

    visualizer.add_scene(config.scenes[0], opacity=0.1)
    
    # smearer = SmearMast3r(config)
    # smeared_chunk = smearer(data)

    # # visualizer.add_points(rearrange(smeared_chunk["verbose"]["coordinates"], "C X Y Z -> (X Y Z) C"))


    # # visualizer.add_from_scene_occ(data)
    #visualizer.add_scene(data["scene_name"], opacity=0.1)


    # # visualizer.show()  
    
    # test collate function: 
    
    collate_fn = transforms_batched.ComposeTransforms(data_config)
    collate_fn.transforms = [transform(data_config, None) for transform in collate_fn.transforms]
    
    occ = collate_fn([zip[3], zip[4], zip[5]])
    
    
    data = zip[5]
    
    data["Y"] = occ["Y"][2]
    data["X"] = occ["X"][2]
    data["verbose"]["data_dict"]["coordinates"]= occ["coordinates"][2]
    #visualizer.add_from_occupancy_dict(data, opacity=0.1, transformed=True)
    #visualizer.add_from_occupancy_dict_as_points(data, opacity=0.1, color="red", with_transform=True)
    visualizer.add_from_image_dict(data["verbose"]["data_dict"])
    visualizer.add_colored_pointcloud_from_data_dict(data)
        #visualizer.add_from_smearing_transform(data)

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