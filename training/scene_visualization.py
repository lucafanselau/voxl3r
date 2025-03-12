from typing import Literal
from loguru import logger
import torch
from tqdm import tqdm
from networks.aggregator_net import AggregatorNet
from networks.surfacenet import SurfaceNet
from training.common import (
    create_datamodule,
    create_dataset,
    load_config_from_checkpoint,
)
from torch.utils.data import DataLoader
from training.default.data import DefaultDataModule
from training.default.module import BaseLightningModule
from training.mast3r.train_aggregator import Config as TrainConfig
from datasets import scene, transforms, transforms_batched

run_name = "kfc9dsju"  # BEST local feat "ohkmg3nr"  # BEST feat based "wxklqj28"
# group = "08_trial_transformer_unet3d"
project_name = "mast3r-3d-experiments"
DataModule = DefaultDataModule
Module = BaseLightningModule  # LightningModuleWithAux  # BaseLightningModule #UNet3DLightningModule
ModelClass = SurfaceNet  # [SurfaceNet, AggregatorNet]
ConfigClass = TrainConfig  # UNet3DConfig
Config = ConfigClass

scene_name = "7b6477cb95"  # scene not found: "acd95847c5", "fb5a96b1a2"


def load_run(run_name, project_name):

    config, path = load_config_from_checkpoint(
        project_name, run_name, ConfigClass=ConfigClass
    )
    config.scenes = [scene_name]
    dataset = create_dataset(config, "val", transform=None)

    samplerConfig = config.model_copy()
    occGridSampler = transforms_batched.ComposeTransforms(samplerConfig)
    module = Module.load_from_checkpoint(
        path, module_config=config, ModelClass=ModelClass, occGridSampler=occGridSampler
    )

    return dataset, module, config


@torch.no_grad()
def save_scene():
    # load run
    dataset, module, _ = load_run(run_name, project_name)
    dataset.prepare_data()

    # get all chunks of scene
    chunks = list(dataset)

    # find the scene dict
    base_dataset = dataset.datasets[0].base_dataset
    scene_dict = base_dataset[base_dataset.get_index_from_scene(scene_name)]

    # get all possible inference chunks
    inference_chunks = []
    base = torch.from_numpy(scene_dict["mesh"].bounds[0].copy())

    mesh_extent = torch.from_numpy(scene_dict["mesh"].extents.copy())
    chunk_extent = base_dataset.data_config.grid_resolution_sample * torch.tensor(
        base_dataset.data_config.grid_size_sample
    )
    num_chunks = torch.ceil((mesh_extent / chunk_extent)).int()

    centers = torch.stack([chunk["chunk_center"] for chunk in chunks])
    threshold = torch.linalg.norm(chunks[0]["chunk_extent"] / 2)

    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            for k in range(num_chunks[2]):
                # find chunk matching
                idx = torch.tensor([i, j, k])

                center = base + idx * chunk_extent + chunk_extent / 2

                # find the closest center
                dist = torch.linalg.norm(centers - center, axis=-1)
                min_dist_idx = torch.argmin(dist)
                if dist[min_dist_idx] > threshold:
                    logger.warning(f"Chunk {idx} is too far from any existing chunk")
                    continue

                added_chunk = chunks[min_dist_idx].copy()
                added_chunk["chunk_center"] = center
                added_chunk["chunk_extent"] = chunk_extent
                inference_chunks.append(added_chunk)

    # create transform and apply to all inference chunks
    transform = transforms.ComposeTransforms(base_dataset.data_config)
    transform.eval()

    inference_chunks = [transform(chunk) for chunk in tqdm(inference_chunks)]

    # collate_fn = transforms_batched.ComposeTransforms(
    #     base_dataset.data_config.model_copy()
    # )
    # collate_fn.transforms = [
    #     transform(base_dataset.data_config, None) for transform in collate_fn.transforms
    # ]
    dataloader = DataLoader(
        inference_chunks,
        batch_size=base_dataset.data_config.batch_size,
        num_workers=base_dataset.data_config.val_num_workers,
        shuffle=True,
        # persistent_workers=True if self.data_config.num_workers > 0 else False,
        generator=torch.Generator().manual_seed(42),
        # collate_fn=collate_fn,
        drop_last=True,
        # pin_memory=True,
    )

    module.eval()
    module.to("cuda")

    def parse_results(results, batch):
        return {
            "Y": results["y"].detach().cpu(),
            "chunk_center": batch["logger"]["origin"].detach().cpu(),
            "pitch": batch["logger"]["pitch"].detach().cpu(),
            "pred": results["pred"].detach().cpu(),
            "loss": results["loss"].detach().cpu(),
            "images": batch["images"],
            "T_cw": batch["logger"]["T_cw"].detach().cpu(),
            "K": batch["logger"]["K"].detach().cpu(),
        }

    results = [
        parse_results(
            module.validation_step(
                {
                    k: v.to(module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                },
                i,
            ),
            batch,
        )
        for i, batch in tqdm(
            enumerate(dataloader),
        )
    ]

    # save results
    torch.save(results, f".visualization/{scene_name}_chunks.pt")


def visualize_scene(treshold=0.5, type: Literal["gt", "pred"] = "pred"):
    import visualization

    dataset, module, config = load_run(run_name, project_name)
    results = torch.load(f".visualization/{scene_name}_chunks.pt")

    visualizer_config = visualization.Config(
        log_dir=".visualization", **config.model_dump()
    )
    visualizer_config.disable_outline = True
    visualizer = visualization.Visualizer(visualizer_config)

    transform_config = config.copy()
    transform_config.transforms = ["SampleCoordinateGrid", "SmearImages"]
    transform = transforms.ComposeTransforms(transform_config)
    transform.eval()

    for result in tqdm(results):
        pred = result["Y"] if type == "gt" else result["pred"]
        center = result["chunk_center"]
        pitch = result["pitch"]

        for i in range(pred.shape[0]):

            # result_transformed = transform(result[i])

            if center[i].numpy()[-1] > 2.0:
                continue
            visualizer.add_occupancy(
                (pred[i][0] > treshold).int(),
                origin=center[i].numpy(),
                pitch=pitch[i].item(),
                opacity=1.0,
            )

    # visualizer.add_mesh(dataset.datasets[0].base_dataset[0]["mesh"].simplify_quadric_decimation(0.95))

    visualizer.export_html("out", timestamp=True)
    visualizer.export_gltf(f"./.visualization/{scene_name}_{type}_without_outline.gltf")

    visualizer = visualization.Visualizer(visualizer_config)
    visualizer.add_mesh(
        dataset.datasets[0].base_dataset[0]["mesh"].simplify_quadric_decimation(0.95)
    )
    visualizer.export_gltf(f"./.visualization/{scene_name}_{type}_scene_mesh.gltf")


def main():
    # save_scene()
    # visualize_scene(type="gt")
    visualize_scene(type="pred")
    return


if __name__ == "__main__":
    main()
