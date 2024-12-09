from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from einops import rearrange
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

from positional_encodings.torch_encodings import PositionalEncoding3D


from datasets.scene import (
    SceneDatasetTransformLoadImages,
)
from experiments.mast3r import load_model
from experiments.mast3r_baseline.module import (
    Mast3rBaselineConfig,
    Mast3rBaselineLightningModule,
)
from datasets.mast3r_chunk_dataset import (
    Mast3rChunkDataset,
    Mast3rChunkDatasetConfig,
)
from datasets.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
from datasets.transforms.projection import project_voxel_grid_to_images_seperate
from utils.chunking import (
    compute_coordinates,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose


config = load_yaml_munch("./utils/config.yaml")


@dataclass
class LocalFeatureGridTransformConfig:
    pe_enabled: bool = False

    add_projected_depth: bool = False
    add_validity_indicator: bool = False
    add_viewing_directing: bool = False

    concatinate_pe: bool = False

    seq_len: int = 4


class LocalFeatureGridTransform:
    """Transform that unwraps the VoxelGridDataset return tuple to only return feature_grid and occupancy_grid"""

    def __init__(
        self,
        config: LocalFeatureGridTransformConfig,
        image_transform=SceneDatasetTransformLoadImages(),
    ):
        self.config = config
        self.image_transform = image_transform
        self.pe = None

    def __call__(
        self, data: Tuple[torch.Tensor, dict], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_dict = data
        occupancy_grid = data_dict["occupancy_grid"]
        data_dict["grid_size"] = torch.from_numpy(data_dict["grid_size"])

        image_dict = {
            Path(key).name: value
            for key, value in zip(data_dict["images"][0], data_dict["images"][1])
        }

        seq_len = len(image_dict.keys())

        # compute the coordinates of each point in shape
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

        res_dict = {
            **data_dict["pairwise_predictions"][0],
            **data_dict["pairwise_predictions"][1],
        }
        chunk_data["images"] = torch.stack(
            [res_dict[f"desc_{image}"] for image in image_dict.keys()]
        )
        chunk_data["camera_params"] = image_dict
        images, transformations, T_cw = self.image_transform.forward(
            chunk_data, images_loaded=True
        )
        images = rearrange(images, "I H W C -> I C H W")

        feature_grid = project_voxel_grid_to_images_seperate(
            coordinates,
            images,
            transformations,
            T_cw,
        )

        local_features, projected_depth, validity_indicator, viewing_direction = (
            feature_grid
        )

        rand_idx = torch.randperm(self.config.seq_len)
        indices_viewing_dir = (
            torch.arange(viewing_direction.shape[0])
            .reshape(-1, viewing_direction.shape[0] // seq_len)[rand_idx]
            .flatten()
        )
        indices_local_features = (
            torch.arange(local_features.shape[0])
            .reshape(-1, local_features.shape[0] // seq_len)[rand_idx]
            .flatten()
        )

        # shuffle the features in first dimension
        local_features = local_features[indices_local_features]
        viewing_direction = viewing_direction[indices_viewing_dir]
        projected_depth = projected_depth[rand_idx]
        validity_indicator = validity_indicator[rand_idx]

        sampled = local_features

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
            local_features.shape[0] // seq_len
            + self.config.add_projected_depth
            + self.config.add_validity_indicator
            + viewing_direction.shape[0] // seq_len * self.config.add_viewing_directing
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

            if self.pe is None:
                self.pe = PositionalEncoding3D(channels).to(sampled.device)

            sampled_reshaped = rearrange(sampled, "C X Y Z -> 1 X Y Z C")
            pe_tensor = self.pe(sampled_reshaped)
            pe_tensor = rearrange(pe_tensor, "1 X Y Z C -> C X Y Z")
            if self.config.concatinate_pe:
                sampled = torch.cat([sampled, pe_tensor], dim=0)
            else:
                sampled = sampled + pe_tensor

        return sampled, occupancy_grid, idx


@dataclass
class Mast3r3DDataConfig(Mast3rChunkDatasetConfig, LocalFeatureGridTransformConfig):
    batch_size: int = 16


class Mast3r3DDataModule(pl.LightningDataModule):
    def __init__(self, data_config: Mast3r3DDataConfig, no_transform: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_config = data_config
        self.image_transform = SceneDatasetTransformLoadImages()
        self.transform = LocalFeatureGridTransform(
            data_config, image_transform=self.image_transform
        )

        self.mast3r_grid_dataset = Mast3rChunkDataset(
            self.data_config, transform=None if no_transform else self.transform
        )

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_in_channels(self):
        return (
            (
                24
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
        self.mast3r_grid_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.mast3r_grid_dataset,
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
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            # pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here
