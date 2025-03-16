from multiprocessing import Pool
from typing import Literal
from einops import rearrange, reduce
from loguru import logger
import torch
from tqdm import tqdm
from networks.aggregator_net import AggregatorNet
from networks.surfacenet import SurfaceNet
from networks.surfacenet_with_attention import SurfaceNetTransformer
from training.common import (
    create_datamodule,
    create_dataset,
    create_dataset_rgb,
    load_config_from_checkpoint,
)
from torch.utils.data import DataLoader
from training.default.data import DefaultDataModule
from training.default.module import BaseLightningModule

#from training.mast3r.train_aggregator import Config as TrainConfig
from training.mast3r.train_pretrained_surfacenet import Config as TrainConfig
from datasets import scene, transforms, transforms_batched

run_name = (
    "8tg2ad9c" #attention # "kfc9dsju"  # BEST local feat "ohkmg3nr"  # BEST feat based "wxklqj28"
)
ckpt = "epoch=7-step=10944" # attnetion: "epoch=7-step=10944"
# group = "08_trial_transformer_unet3d"
project_name = "mast3r-3d-experiments"
DataModule = DefaultDataModule
Module = BaseLightningModule  # LightningModuleWithAux  # BaseLightningModule #UNet3DLightningModule
ModelClass = SurfaceNetTransformer  # [SurfaceNet, AggregatorNet]
ConfigClass = TrainConfig  # UNet3DConfig
Config = ConfigClass

scene_name = "7b6477cb95"  # scene not found: "acd95847c5", "fb5a96b1a2"
multipliers_chunk_center = 8 # can be 1,2,4,8 ... every number that can be used to devide 32 without leaving a remainder

# dataloader config
batch_size = 128
num_workers = 11

def load_run(run_name, project_name, load_module=True, rgb=False):

    config, path = load_config_from_checkpoint(
        project_name, run_name, ConfigClass=ConfigClass, checkpoint_name=ckpt
    )
    config.scenes = [scene_name]
    if rgb:
        dataset = create_dataset_rgb(config, "val", transform=None)
    else:
        dataset = create_dataset(config, "val", transform=None)

    samplerConfig = config.model_copy()
    occGridSampler = transforms_batched.ComposeTransforms(samplerConfig)

    if load_module:
        module = Module.load_from_checkpoint(
            path,
            module_config=config,
            ModelClass=ModelClass,
            occGridSampler=occGridSampler,
        )
        module.eval()
        module.to("cuda")

        return dataset, module, config
    else:
        return dataset, occGridSampler, config


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
                
                for n1 in range(multipliers_chunk_center):
                    for n2 in range(multipliers_chunk_center):
                        for n3 in range(multipliers_chunk_center):
                            mulitplier = torch.Tensor([n1, n2, n3])
                            chunk_offset = mulitplier*chunk_extent/2

                            center = base + idx * chunk_extent + chunk_offset

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

    #inference_chunks = [transform(chunk) for chunk in tqdm(inference_chunks)]
    new_inference_chunks = []
    with Pool(num_workers) as p:
        with tqdm(total=len(inference_chunks), position=0, leave=True) as pbar:
            # Use imap_unordered to process chunks in parallel
            for transformed_chunk in p.imap_unordered(transform, inference_chunks):
                new_inference_chunks.append(transformed_chunk)
                pbar.update(1)
                
    inference_chunks = new_inference_chunks

    # collate_fn = transforms_batched.ComposeTransforms(
    #     base_dataset.data_config.model_copy()
    # )
    # collate_fn.transforms = [
    #     transform(base_dataset.data_config, None) for transform in collate_fn.transforms
    # ]
    dataloader = DataLoader(
        inference_chunks,
        batch_size=batch_size,
        num_workers=num_workers,
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
            "multipliers_chunk_center": multipliers_chunk_center,
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
    torch.save(results, f".visualization/{scene_name}_chunks_multiplier_{multipliers_chunk_center}.pt")


def visualize_scene(threshold=0.5, type: Literal["gt", "pred"] = "pred", color=False):
    import visualization

    dataset, occGridSampler, config = load_run(
        run_name, project_name, load_module=False, rgb=color
    )
    dataset.prepare_data()
    results = torch.load(f".visualization/{scene_name}_chunks_multiplier_{multipliers_chunk_center}.pt")

    visualizer_config = visualization.Config(
        log_dir=".visualization", **config.model_dump()
    )
    visualizer_config.disable_outline = False
    visualizer = visualization.Visualizer(visualizer_config)

    transform_config = config.copy()
    transform_config.transforms = ["SampleCoordinateGrid"]
    transform = transforms.ComposeTransforms(transform_config)
    transform.eval()

    smear = transforms.BaseSmear(config.model_copy())

    image_chunks = [chunk for chunk in tqdm(dataset) if "images_tensor" in chunk]
    centers = torch.stack([chunk["chunk_center"] for chunk in image_chunks])
    transformed_chunks = []
    # loop over all results and find the closest image_chunk
    for result in tqdm(results):
        for i in range(result["Y"].shape[0]):
            dist = torch.linalg.norm(centers - result["chunk_center"][i], axis=-1)
            min_dist_idx = torch.argmin(dist)
            best_chunk = image_chunks[min_dist_idx]
            chunk = best_chunk.copy()
            chunk.update(
                {
                    "chunk_center": result["chunk_center"][i],
                    "Y": result["Y"][i] if type == "gt" else result["pred"][i],
                    "pitch": result["pitch"][i],
                    "T_cw": result["T_cw"][i],
                    "K": result["K"][i],
                    "multipliers_chunk_center": result["multipliers_chunk_center"],
                }
            )
            transformed_chunks.append(chunk)

    transformed_chunks = [transform(chunk) for chunk in tqdm(transformed_chunks)]

    for chunk in tqdm(transformed_chunks):
        if chunk["chunk_center"][-1] > 2.0:
            logger.warning(f"Chunk {chunk['chunk_center']} is in the ceiling, skipping")
            continue

        grid_color = None
        if color:
            transformation = chunk["K"].float() @ chunk["T_cw"][..., :3, :]

            # make images red
            images = chunk["images_tensor"].clone()

            grid_color, _ = smear.smear_images(
                rearrange(images, "B P ... -> (B P) ..."),
                transformation,
                chunk["T_cw"],
                chunk["coordinates"],
            )
            grid_color = reduce(grid_color, "B ... -> ...", "mean")
            # grid_color = grid_color[0]
            # add 1s in the first dimension (rgb -> rgba)
            # grid_color = torch.cat(
            #     [grid_color, torch.ones_like(grid_color[:1, ...])], dim=0
            # )
        
        # include multiplier to crop chunks
        multipliers_chunk_center = chunk["multipliers_chunk_center"]
        chunk_extent = (chunk["Y"][0].shape[-1] // multipliers_chunk_center)
        start_idx = chunk_extent//2
        end_idx = start_idx + chunk_extent
        chunk_cropped = chunk["Y"][0][:, start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        grid_color_cropped = grid_color[:, start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        visualizer.add_occupancy(
            (chunk_cropped > 0.5).int(),
            origin=chunk["chunk_center"].numpy(),
            pitch=chunk["pitch"].numpy(),
            opacity=1.0,
            color=grid_color_cropped,
        )
        # visualizer.add_occupancy(
        #     (chunk["Y"][0] > 0.5).int(),
        #     origin=chunk["chunk_center"].numpy(),
        #     pitch=chunk["pitch"].numpy(),
        #     opacity=1.0,
        #     color=grid_color,
        # )

    # for result in tqdm(results):
    #     pred = result["Y"] if type == "gt" else result["pred"]
    #     center = result["chunk_center"]
    #     pitch = result["pitch"]

    #     for i in range(pred.shape[0]):
    #         if color:
    #             # get the closest image_chunk
    #             dist = torch.linalg.norm(centers - center[i], axis=-1)
    #             min_dist_idx = torch.argmin(dist)
    #             if dist[min_dist_idx] > threshold:
    #                 logger.warning(
    #                     f"Chunk {result['chunk_center']} is too far from any existing chunk"
    #                 )
    #                 continue

    #             best_chunk = image_chunks[min_dist_idx]

    #         # result_transformed = transform(result[i])

    #         if center[i].numpy()[-1] > 2.0:
    #             continue
    #         visualizer.add_occupancy(
    #             (pred[i][0] > treshold).int(),
    #             origin=center[i].numpy(),
    #             pitch=pitch[i].item(),
    #             opacity=1.0,
    #         )

    # visualizer.add_mesh(dataset.datasets[0].base_dataset[0]["mesh"].simplify_quadric_decimation(0.95))

    visualizer.export_html("out", timestamp=True)
    visualizer.export_gltf(f"./.visualization/{scene_name}_{type}_without_outline.gltf")

    visualizer = visualization.Visualizer(visualizer_config)
    visualizer.add_mesh(
        dataset.datasets[0].base_dataset[0]["mesh"].simplify_quadric_decimation(0.95)
    )
    visualizer.export_gltf(f"./.visualization/{scene_name}_{type}_scene_mesh.gltf")


def main():
    save_scene()
    #visualize_scene(type="gt", color=True)
    visualize_scene(type="pred", color=True)
    return


if __name__ == "__main__":
    main()
