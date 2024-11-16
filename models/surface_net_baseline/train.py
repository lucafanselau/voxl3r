from pathlib import Path
import numpy as np
import torch
from torchvision.io import read_image

from dataset import SceneDataset, SceneDatasetTransformToTorch
from einops import rearrange
from models.surface_net_baseline.model import SimpleOccNetConfig
from models.surface_net_baseline.module import LRConfig, OccSurfaceNet, OptimizerConfig
from models.surface_net_baseline.data import (
    OccSurfaceNetDatamodule,
    OccSurfaceNetDataset,
    custom_collate_fn,
    project_points_to_images,
)
from utils.data_parsing import load_yaml_munch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.visualize import plot_voxel_grid, visualize_mesh


cfg = load_yaml_munch(Path("utils") / "config.yaml")
visualize = False
visualize_whole_scene = True


def visualize_unprojection(data):
    transform = SceneDatasetTransformToTorch("cuda")
    image_names, camera_params_list = data["images"]
    images, transformations, points, gt = transform.forward(data)
    # and normalize images
    images = images / 255.0

    X = project_points_to_images(points, images, transformations)

    rgb_list = rearrange(X, "p (i c) -> p i c", c=3)
    mask = rgb_list != -1
    denom = torch.sum(torch.sum(mask, -1) / 3, -1)
    rgb_list[rgb_list == -1.0] = 0.0
    rgb_list_pruned = rgb_list[denom != 0]
    points_pruned = points[denom != 0]
    occ = gt[denom != 0]
    denom = denom[denom != 0]
    rgb_list_avg = torch.sum(rgb_list_pruned, dim=1) / denom.unsqueeze(-1).repeat(1, 3)

    visualize_mesh(
        data["mesh"],
        point_coords=points_pruned.cpu().numpy(),
        images=image_names,
        camera_params_list=camera_params_list,
        heat_values=occ.cpu().numpy(),
        rgb_list=rgb_list_avg.cpu().numpy(),
    )


def visualize_unprojection_whole_scene(base_dataset, scene):
    idx = base_dataset.get_index_from_scene(scene)
    dataset = OccSurfaceNetDataset(
        base_dataset, scene, p_batch_size=4096, max_seq_len=16, len_chunks=5
    )
    dataset.prepare_data()

    batch_size = 4

    transform = SceneDatasetTransformToTorch(cfg.device)
    all_points = []
    all_occ = []
    all_rgb = []
    for i in range(len(dataset) // batch_size):

        X, gt, points = custom_collate_fn(
            [dataset[i * batch_size + j] for j in range(batch_size)]
        )
        gt = gt.squeeze()

        rgb_list = rearrange(X, "p (i c) -> p i c", c=3)
        mask = rgb_list != -1
        denom = torch.sum(torch.sum(mask, -1) / 3, -1)
        rgb_list[rgb_list == -1.0] = 0.0
        rgb_list_pruned = rgb_list[denom != 0]
        points_pruned = points[denom != 0]
        occ = gt[denom != 0]
        denom = denom[denom != 0]
        rgb_list_avg = torch.sum(rgb_list_pruned, dim=1) / denom.unsqueeze(-1).repeat(
            1, 3
        )
        all_points.append(points_pruned)
        all_occ.append(occ)
        all_rgb.append(rgb_list_avg)
    mesh = (
        Path(base_dataset.data_dir)
        / base_dataset.scenes[idx]
        / "scans"
        / "mesh_aligned_0.05.ply"
    )
    all_points = torch.cat(all_points)
    all_occ = torch.cat(all_occ)
    all_rgb = torch.cat(all_rgb)
    visualize_mesh(
        mesh,
        point_coords=all_points.cpu().numpy(),
        heat_values=all_occ.cpu().numpy(),
        rgb_list=all_rgb.cpu().numpy(),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    max_seq_len = 8
    scene_dataset = SceneDataset(
        data_dir="/home/luca/mnt/data/scannetpp/data",
        camera="iphone",
    )

    if visualize:
        visualize_unprojection(scene_dataset, scene="0cf2e9402d")
    if visualize_whole_scene:
        visualize_unprojection_whole_scene(scene_dataset, scene="0cf2e9402d")

    # datamodule = OccSurfaceNetDatamodule(scene_dataset, "0cf2e9402d", batch_size=2048, max_sequence_length=max_seq_len)
    datamodule = OccSurfaceNetDatamodule(
        scene_dataset,
        "0cf2e9402d",
        batch_size=8,
        p_in_batch=4096,
        max_seq_len=max_seq_len,
        channels=0,
        # image_name="frame_001000.jpg"
    )

    max_epochs = 100

    # model = OccSurfaceNet.load_from_checkpoint(".lightning/occ-surface-net/surface-net-baseline/wjcst3w3/checkpoints/epoch=340-step=8866.ckpt")
    # Initialize OccSurfaceNet
    model = OccSurfaceNet(
        SimpleOccNetConfig(
            input_dim=max_seq_len * 3 + datamodule.channels, hidden=[2048, 2048, 2048]
        ),
        OptimizerConfig(learning_rate=4e-4, weight_decay=0.0),
        LRConfig(T_max=max_epochs),
    )
    logger = WandbLogger(
        project="surface-net-baseline", save_dir="./.lightning/occ-surface-net"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # custom ModelCheckpoint
    # Save top3 models wrt precision
    filename = "{epoch}-{step:.2f}"
    callbacks = [
        ModelCheckpoint(
            # -{val_accuracy:.2f}
            filename=filename + f"-{{val_{name}:.2f}}",
            monitor=f"val_{name}",
            save_top_k=3,
            mode=mode,
        )
        for [name, mode] in [["accuracy", "max"], ["loss", "min"]]
    ]

    # Save the model every 5 epochs
    every_five_epochs = ModelCheckpoint(
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
    )

    # last run: 2q2io4h0
    print("Running training...")
    torch.compile(model)
    print("Compiled model")
    trainer = Trainer(
        max_epochs=max_epochs,
        # Used to limit the number of batches for testing and initial overfitting
        # limit_train_batches=8,
        limit_val_batches=2,
        # Logging stuff
        log_every_n_steps=4,
        callbacks=[*callbacks, every_five_epochs, lr_monitor],
        logger=logger,
        profiler="simple",
        # Performance stuff
        precision="bf16-mixed",
        default_root_dir="./.lightning/occ-surface-net",
    )
    trainer.fit(model, datamodule=datamodule)

    # Get the run, based on the checkpoints
    base_path = Path(callbacks[0].best_model_path).parents[1]
    result = base_path / "best_ckpts.pt"
    result_dict = {}
    result_dict["best_model_val_accuracy"] = callbacks[0].best_model_path
    result_dict["best_model_val_loss"] = callbacks[1].best_model_path
    result_dict["last_model_path"] = every_five_epochs.best_model_path
    torch.save(result_dict, result)

    print("Running test on best model...")
    # this should be with regard to the validation set
    # trainer.test(ckpt_path=callbacks[0].best_model_path, dataloaders=datamodule)
    dict = torch.load(
        "./.lightning/occ-surface-net/surface-net-baseline/w8mnl00u/best_ckpts.pt"
    )
    # model_path = dict["best_model_val_accuracy"]
    model_path = callbacks[0].best_model_path

    # trainer.test(model_path, dataloaders=datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")

    model = OccSurfaceNet.load_from_checkpoint(model_path)
    test_dict = model.test_visualize(datamodule.test_dataloader())

    gt = torch.cat(test_dict["gt"])
    points = torch.cat(test_dict["points"])
    y = torch.sigmoid(torch.cat(test_dict["out"]))
    y[y < 0.5] = 0.0
    y[y > 0.5] = 1.0

    mesh = (
        Path(scene_dataset.data_dir)
        / scene_dataset.scenes[datamodule.scene_idx_test]
        / "scans"
        / "mesh_aligned_0.05.ply"
    )
    # visualize_mesh(mesh, point_coords=points.cpu().numpy(), heat_values=y.cpu().numpy())
    plot_voxel_grid(
        points.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        ref_mesh=mesh,
        resolution=scene_dataset.resolution,
    )
