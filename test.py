from datasets import chunk, scene, transforms
from utils.data_parsing import load_yaml_munch


class DataConfig(chunk.occupancy.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig):
    pass




base_dataset = scene.Dataset(config)
image_dataset = chunk.image.Dataset(config, base_dataset)

zip = chunk.zip.ZipChunkDataset([
    image_dataset,
    chunk.occupancy.Dataset(config, base_dataset, image_dataset),
    chunk.mast3r.Dataset(config, base_dataset, image_dataset),
], transform=transforms.SmearMast3r(config))

zip.prepare_data()

zip[0]

for i in zip:
    print(i)