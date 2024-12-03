from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from einops import rearrange
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

from positional_encodings.torch_encodings import PositionalEncoding3D


from dataset import (
    SceneDatasetTransformLoadImages,
)
from models.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
from models.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from utils.chunking import (
    compute_coordinates,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose


config = load_yaml_munch("./utils/config.yaml")


@dataclass
class ColorFeatureGridTransformConfig:
    pe_enabled: bool = False

    add_projected_depth: bool = False
    add_validity_indicator: bool = False
    add_viewing_directing: bool = False

    concatinate_pe: bool = False

    seq_len: int = 4


class ColorFeatureGridTransform:
    """Transform that unwraps the VoxelGridDataset return tuple to only return feature_grid and occupancy_grid"""

    def __init__(self, config: ColorFeatureGridTransformConfig):
        self.config = config
        self.image_transform = SceneDatasetTransformLoadImages()

    def __call__(
        self, data: Tuple[torch.Tensor, dict], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        occupancy_grid, data_dict = data

        image_folder = Path(data_dict["images"][0][0]).parents[0]
        image_dict = {
            Path(key).name: value
            for key, value in zip(data_dict["images"][0], data_dict["images"][1])
        }

        # compute the coordinates of each point in shace
        image_name = data_dict["image_name_chunk"]
        T_cw = image_dict[image_name]["T_cw"]
        _, _, T_wc = invert_pose(T_cw[:3, :3], T_cw[:3, 3])
        coordinates = compute_coordinates(
            occupancy_grid,
            data_dict["center"],
            data_dict["resolution"],
            data_dict["grid_size"][0],
            to_world_coordinates=T_wc,
        )
        coordinates = torch.from_numpy(coordinates).float().to(occupancy_grid.device)

        # transform images into space
        chunk_data = {}
        chunk_data["image_names"] = [
            image_folder / image for image in image_dict.keys()
        ]
        chunk_data["camera_params"] = image_dict
        images, transformations, T_cw = self.image_transform.forward(chunk_data)
        feature_grid = project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
        )
        rgb_features, projected_depth, validity_indicator, viewing_direction = (
            feature_grid
        )

        rand_idx = torch.randperm(self.config.seq_len)
        indices = torch.arange(rgb_features.shape[0]).reshape(-1, 3)[rand_idx].flatten()

        # shuffle the features in first dimension
        rgb_features = rgb_features[indices]
        viewing_direction = viewing_direction[indices]
        projected_depth = projected_depth[rand_idx]
        validity_indicator = validity_indicator[rand_idx]

        sampled = rgb_features

        if self.config.add_projected_depth:
            sampled = torch.cat([sampled, projected_depth])

        if self.config.add_validity_indicator:
            sampled = torch.cat([sampled, validity_indicator])

        if self.config.add_viewing_directing:
            sampled = torch.cat([sampled, viewing_direction])

        fill_value = -1.0

        C, W, H, D = sampled.shape

        # TODO: make this dynamic

        num_of_channels = self.config.seq_len * (
            3
            + self.config.add_projected_depth
            + self.config.add_validity_indicator
            + 3 * self.config.add_viewing_directing
        )

        if num_of_channels - C:
            sampled = torch.cat(
                [
                    sampled,
                    torch.full(
                        (num_of_channels - C, W, H, D),
                        fill_value,
                    ).to(sampled.device),
                ],
                dim=-1,
            )

        # give positional envoding even though values can be just filled with -1?
        if self.config.pe_enabled:
            channels = sampled.shape[0]
            global pe
            if pe is None:
                pe = PositionalEncoding3D(channels).to(sampled.device)

            sampled_reshaped = rearrange(sampled, "C X Y Z -> 1 X Y Z C")
            pe_tensor = pe(sampled_reshaped)
            pe_tensor = rearrange(pe_tensor, "1 X Y Z C -> C X Y Z")
            if self.config.concatinate_pe:
                sampled = torch.cat([sampled, pe_tensor], dim=0)
            else:
                sampled = sampled + pe_tensor

        return sampled, occupancy_grid, idx


class UnwrapVoxelGridTransform:
    """Transform that unwraps the VoxelGridDataset return tuple to only return feature_grid and occupancy_grid"""

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor, dict], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_grid, occupancy_grid, _ = data

        return feature_grid, occupancy_grid, idx


@dataclass
class SurfaceNet3DDataConfig(OccChunkDatasetConfig, ColorFeatureGridTransformConfig):
    batch_size: int = 16


class SurfaceNet3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config: SurfaceNet3DDataConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_config = data_config
        self.transform = ColorFeatureGridTransform(data_config)

        self.grid_dataset = OccChunkDataset(self.data_config, transform=self.transform)

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_in_channels(self):
        return (
            (
                3
                + self.data_config.add_projected_depth
                + self.data_config.add_validity_indicator
                + self.data_config.add_viewing_directing * 3
            )
            * self.data_config.seq_len
            * (1 + self.data_config.concatinate_pe)
        )

        # For now the positional encoding is "added" in the projection
        # + (
        #    self.data_config.pe_channels if self.data_config.pe_enabled else 0
        # )

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        self.grid_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.grid_dataset,
            [0.6, 0.2, 0.2],
            generator=torch.Generator().manual_seed(42),
        )

        # Assign datasets based on stage
        if stage == "fit" or stage is None:
            pass  # datasets already assigned

        if stage == "test" or stage is None:
            pass  # test dataset already assigned

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here
