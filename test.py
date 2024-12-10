from datasets import chunk, scene, transforms
from utils.data_parsing import load_yaml_munch
from visualization import Visualizer, Config


class DataConfig(chunk.occupancy.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig):
    pass


if __name__ == "__main__":
    # first load data_config
    config = DataConfig.load_from_files([
            "./config/data/base.yaml",
            "./config/data/mast3r_scenes.yaml"
        ], {
            "skip_prepare": True,
        })
    

    base_dataset = scene.Dataset(config)
    image_dataset = chunk.image.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy.Dataset(config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(config, base_dataset, image_dataset),
    ])

    zip.prepare_data()

    data = zip[0]

    visualizer = Visualizer(Config(log_dir=".visualization", data_dir=config.data_dir))

    visualizer.add_scene(data["scene_name"])
    visualizer.add_from_occupancy_dict(data)
    visualizer.add_from_image_dict(data)

    visualizer.show()

    # visualizer.export_html("out.html")
