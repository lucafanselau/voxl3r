import os
from pathlib import Path
import argparse
from einops import rearrange
import torch
from lightning import Trainer

from experiments.surface_net_3d.model import (
    LitSurfaceNet3D,
    LitSurfaceNet3DConfig,
    SurfaceNet3DConfig,
)
from experiments.surface_net_3d.data import (
    SurfaceNet3DDataConfig,
    SurfaceNet3DDataModule,
)
from experiments.surface_net_3d.projection import get_3d_pe
from experiments.surface_net_3d.visualize import (
    VoxelVisualizerConfig,
    calculate_average_color,
    visualize_voxel_grids,
)
from utils.data_parsing import load_yaml_munch

config = load_yaml_munch("./utils/config.yaml")


def visualize_run(
    run_name: str,
    idx_train: int = 0,
    idx_val: int = 0,
    show: list = ["train", "val", "test"],
    data_config: SurfaceNet3DDataConfig = SurfaceNet3DDataConfig(
        data_dir=config.data_dir, batch_size=1
    ),
):

    # Load best checkpoints
    ckpt_dir = Path("./.lightning/surface-net-3d/surface-net-3d") / run_name
    # check if best_ckpts.pt exists
    if (ckpt_dir / "best_ckpts.pt").exists():
        best_ckpts = torch.load(ckpt_dir / "best_ckpts.pt")
        model_path = best_ckpts["best_model_val_accuracy"]
    else:
        # fallback to last checkpoint
        print("No best_ckpts.pt found, falling back to last checkpoint")
        model_path = ckpt_dir / "checkpoints" / "last.ckpt"

    # Load model from best validation accuracy checkpoint
    model = LitSurfaceNet3D.load_from_checkpoint(model_path)
    model.eval()

    # Initialize datamodule
    datamodule = SurfaceNet3DDataModule(data_config=data_config, transform=None)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")

    # Create visualization config
    vis_config = VoxelVisualizerConfig(
        opacity=1,
        show_edges=False,
        cmap="viridis",
        window_size=(1600, 1200),
        camera_position=(100, 100, 100),
        grid_arrangement=(
            2,
            len(show),
        ),  # Changed to 2 rows (GT and pred) and 3 columns (splits)
        spacing_factor=1.2,
        show_labels=True,
        label_size=14,
    )

    # Get one batch from each split
    train_batch = datamodule.train_dataset[
        idx_train
    ]  # next(iter(datamodule.train_dataloader()))
    val_batch = datamodule.val_dataset[
        idx_val
    ]  # next(iter(datamodule.val_dataloader()))
    test_batch = datamodule.test_dataset[0]  # next(iter(datamodule.test_dataloader()))

    # Process batches
    def process_batch(batch):
        features, occupancy, data_dict = datamodule.transfer_batch_to_device(
            batch, config.device, 0
        )
        features = features.unsqueeze(0)
        with torch.no_grad():
            pred_occupancy = model(features)
            # apply sigmoid
            pred_occupancy = torch.sigmoid(pred_occupancy)
        return occupancy.unsqueeze(0), pred_occupancy, data_dict, features

    # Get predictions for each split
    train_gt, train_pred, train_data_dict, train_features = process_batch(train_batch)
    val_gt, val_pred, val_data_dict, val_features = process_batch(val_batch)
    test_gt, test_pred, test_data_dict, test_features = process_batch(test_batch)

    # do average color channels

    # Combine all predictions and ground truths into one visualization
    tensors_to_cat = []
    if "train" in show:
        tensors_to_cat.append(train_gt[0:1])
    if "val" in show:
        tensors_to_cat.append(val_gt[0:1])
    if "test" in show:
        tensors_to_cat.append(test_gt[0:1])
    if "train" in show:
        tensors_to_cat.append(train_pred[0:1])
    if "val" in show:
        tensors_to_cat.append(val_pred[0:1])
    if "test" in show:
        tensors_to_cat.append(test_pred[0:1])

    combined_occupancy = torch.cat(tensors_to_cat, dim=0)

    # mask only for true values
    eps = 0.5
    mask = (combined_occupancy > eps).squeeze(1).cpu()

    loaded = torch.load(model_path)
    # let's assume we have a data_config in the checkpoint
    # check if data_config is in the checkpoint
    if "data_config" in loaded["hyper_parameters"]:
        print("Found data_config in checkpoint")
        data_config: SurfaceNet3DDataConfig = loaded["hyper_parameters"]["data_config"]
    else:
        print("No data_config found in checkpoint, using default")
        data_config = data_config
    seq_len = data_config.seq_len

    # single tensor with all features
    tensors_to_cat = []
    if "train" in show:
        tensors_to_cat.append(train_features)
    if "val" in show:
        tensors_to_cat.append(val_features)
    if "test" in show:
        tensors_to_cat.append(test_features)
    all_features = torch.cat(tensors_to_cat, dim=0)

    all_features = rearrange(all_features, "b (s f) d h w -> b s f d h w", s=seq_len)
    # only the first 3 feature channels are color values
    all_features_rgb = all_features[:, :, :3, :, :]

    # do average color channels, but only for the gt values
    # make all of occupancy 3 channels (for rgb)
    combined_occupancy = combined_occupancy.repeat(1, 3, 1, 1, 1)
    combined_occupancy[: len(show)] = calculate_average_color(
        all_features_rgb, fill_value=-1
    )

    # Create labels for both ground truth and predictions

    print(f"First image of training is: {train_data_dict["images"][0][0]}")
    print(
        f'Chunk size was {train_data_dict["chunk_size"]}, resolution was {train_data_dict["resolution"]}, center was {train_data_dict["center"]}'
    )
    print(f"Grid size was {train_data_dict["grid_size"][0]}")
    print(f"First image of validation is: {val_data_dict["images"][0][0]}")
    print(
        f"Chunk size was {val_data_dict["chunk_size"]}, resolution was {val_data_dict["resolution"]}, center was {val_data_dict["center"]}"
    )
    print(f"First image of test is: {test_data_dict["images"][0][0]}")
    print(
        f"Chunk size was {test_data_dict["chunk_size"]}, resolution was {test_data_dict["resolution"]}, center was {test_data_dict["center"]}"
    )
    labels = (
        None  # ["Train GT", "Val GT", "Test GT", "Train Pred", "Val Pred", "Test Pred"]
    )

    # Save visualization
    save_path = ckpt_dir / "occupancy_comparison.png"
    visualize_voxel_grids(
        combined_occupancy.detach().cpu(),  # Repeat channels for RGB
        config=vis_config,
        labels=labels,
        mask=mask,
        save_path=str(save_path),
    )
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("run_name", type=str, help="Name of the training run")
    # args = parser.parse_args()

    ckpt_folder = list(Path("./.lightning/surface-net-3d/surface-net-3d/").glob("*"))
    ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
    last_ckpt_folder = ckpt_folder[-1]
    run_name = last_ckpt_folder.stem
    print(f"Last training is {run_name}")
    data_config = SurfaceNet3DDataConfig(
        data_dir=config.data_dir,
        batch_size=16,
        num_workers=11,
        scenes=load_yaml_munch(Path("./data") / "dslr_undistort_config.yml").scene_ids,
    )

    visualize_run(run_name, 120, 0, show=["train"], data_config=data_config)
