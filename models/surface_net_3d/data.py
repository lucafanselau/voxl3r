from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Tuple, List

from beartype import beartype
from einops import rearrange
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, Dataset
from jaxtyping import Float, Int, Bool, jaxtyped
from torch import Tensor
import numpy as np
from tqdm import tqdm
import trimesh

from dataset import SceneDataset, SceneDatasetTransformLoadImages, SceneDatasetTransformToTorch
from models.surface_net_baseline.data import project_points_to_images
from utils.chunking import create_chunk, mesh_2_local_voxels, mesh_2_voxels
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose
from utils.visualize import visualize_mesh
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")


@dataclass
class SurfaceNet3DDataConfig:
    data_dir: str = "datasets/scannetpp/data"
    batch_size: int = 32
    num_workers: int = 4
    camera: str = "iphone"
    train_val_split: float = 0.9
    scenes: Optional[List[str]] = field(default_factory=lambda: list(["0cf2e9402d"]))
    grid_resolution: float = 0.02
    grid_size: Int[np.ndarray, "3"] = field(default_factory=lambda: np.array([128,128,128]))
    pe_enabled: bool = False
    pe_channels: int = 16
    seq_len: int = 16

    force_prepare: bool = False


class VoxelGridDataset(Dataset):
    def __init__(self, base_dataset: SceneDataset, data_config: SurfaceNet3DDataConfig):
        self.base_dataset = base_dataset
        self.data_config = data_config
        self.transform = SceneDatasetTransformLoadImages()
        self.file_names = None

    

    @jaxtyped(typechecker=beartype)
    def convert_scene_to_grids(
        self,
        base_dataset_dict: dict
    ) -> Generator[
        dict, None, None
    ]:

        mesh = base_dataset_dict["mesh"]
        image_path = base_dataset_dict["path_images"]
        camera_params = base_dataset_dict["camera_params"]
        scene_name = base_dataset_dict["scene_name"]

        # important stuff from the config
        resolution = self.data_config.grid_resolution
        grid_size = self.data_config.grid_size
        seq_len = self.data_config.seq_len
        pe_enabled = self.data_config.pe_enabled
        pe_channels = self.data_config.pe_channels
        

        image_names = list(camera_params.keys())

        chunk_size = resolution * grid_size.astype(np.float32)
        center = np.array([0.0, 0.0, (chunk_size[2] / 2) * 1.25])
            
        print("Preparing chunks for training:")
        for i in tqdm(range((len(image_names) // seq_len))):

            image_name = image_names[i * seq_len]
            data_chunk = create_chunk(
                mesh.copy(),
                image_name,
                camera_params,
                max_seq_len=seq_len,
                image_path=image_path,
                size = chunk_size,
                center= center
            )

            transformation = data_chunk["camera_params"][image_name]["T_cw"]
            _, _, T_wc = invert_pose(
                transformation[:3, :3], transformation[:3, 3]
            )
            
            voxel_grid, coordinate_grid, occupancy_grid = mesh_2_local_voxels(
                data_chunk["mesh"], center=center, pitch=resolution, final_dim=grid_size[0], to_world_coordinates=T_wc
            )

            if False:
                coordinates = rearrange(coordinate_grid, "c x y z -> (x y z) c")
                visualize_mesh(data_chunk["backtransformed"], point_coords=coordinates[occupancy_grid.flatten()==1])


            images, transformations = self.transform.forward(data_chunk)
            coordinates = torch.from_numpy(coordinate_grid).float().to(images.device)
            occupancy_grid = torch.from_numpy(occupancy_grid).to(images.device)
            
            images = images / 255.0

            # unprojection
            points = rearrange(coordinates, "c x y z -> (x y z) c")

            X = project_points_to_images(points, images, transformations, add_positional_encoding=pe_enabled, channels=pe_channels)

            feature_grid = rearrange(X, "(x y z) c -> c x y z", x=grid_size[0], y=grid_size[1], z=grid_size[2])

            if False:
                gt = rearrange(occupancy_grid, "x y z -> (x y z) 1")
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
                    data_chunk["backtransformed"],
                    point_coords=points_pruned.cpu().numpy(),
                    images=data_chunk["image_names"],
                    camera_params_list=data_chunk["camera_params"].values(),
                    heat_values=occ.flatten().cpu().numpy(),
                    rgb_list=rgb_list_avg.cpu().numpy(),
                )

            # rearrange occupancy_grid to 1 W H D
            occupancy_grid = rearrange(occupancy_grid, "x y z -> 1 x y z")

            result_dict = {
                "name": scene_name,
                "resolution": resolution,
                "grid_size": grid_size,
                "chunk_size": chunk_size,
                "center": center,
                "training_data": (feature_grid, occupancy_grid),
                "image_name_chunk": image_name,
                "pe_channels": pe_channels,
                "images": (
                    [str(name) for name in data_chunk["image_names"]],
                    list(data_chunk["camera_params"].values()),
                ),
            }

            yield result_dict
            
    def prepare_scene(self, scene_name):
        camera = self.data_config.camera
        grid_res = self.data_config.grid_resolution
        grid_size = self.data_config.grid_size
        seq_len = self.data_config.seq_len
        pe_enabled = self.data_config.pe_enabled
        pe_channels = self.data_config.pe_channels

        # check if preprared data exists otherwise otherwise continue
        data_dir = Path(self.data_config.data_dir) / scene_name / "prepared_grids" / camera / f"seq_len_{seq_len}_res_{grid_res}_size_{grid_size}_pe_enabled_{pe_channels if pe_enabled else "None"}"
        if data_dir.exists() and not self.data_config.force_prepare:
            return
        
        idx = self.base_dataset.get_index_from_scene(scene_name)
        
        for i, scene_dict in enumerate(self.convert_scene_to_grids(self.base_dataset[idx])):
            image_name_chunk = scene_dict["image_name_chunk"]
            data_dir = Path(self.data_config.data_dir) / scene_name / "prepared_grids" / camera / f"seq_len_{seq_len}_res_{grid_res}_size_{grid_size}_pe_enabled_{pe_channels if pe_enabled else "None"}"
            if data_dir.exists() == False:
                data_dir.mkdir(parents=True)
            
            torch.save(scene_dict, data_dir / f"{i}_{image_name_chunk}.pt")

    def prepare_data(self):
        scenes = self.data_config.scenes if self.data_config.scenes is not None else self.base_dataset.scenes
        with Pool(self.data_config.num_workers) as p:
            p.map(self.prepare_scene, scenes)

        self.load_paths()


    def load_paths(self):
        camera = self.data_config.camera
        grid_res = self.data_config.grid_resolution
        grid_size = self.data_config.grid_size
        seq_len = self.data_config.seq_len
        pe_enabled = self.data_config.pe_enabled
        pe_channels = self.data_config.pe_channels

        self.file_names = {}

        for scene_name in self.data_config.scenes if self.data_config.scenes is not None else self.base_dataset.scenes:
            data_dir = Path(self.data_config.data_dir) / scene_name / "prepared_grids" / camera / f"seq_len_{seq_len}_res_{grid_res}_size_{grid_size}_pe_enabled_{pe_channels if pe_enabled else "None"}"

            self.file_names[scene_name] = list(data_dir.iterdir())
            

    def __getitem__(
        self, idx
    ) -> Tuple[
        Float[Tensor, "channels height width depth"], Bool[Tensor, "height width depth"], dict
    ]:
        if self.file_names is None:
            raise ValueError("No files loaded. Perhaps you forgot to call prepare_data()?")

        # flattened self.file_names
        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]

        data = torch.load(file)
        feature_grid, occupancy_grid = data["training_data"]
        return feature_grid, occupancy_grid, data
    
    def __len__(self):
        if self.file_names is None:
            raise ValueError("No files loaded. Perhaps you forgot to call prepare_data()?")
        return sum([len(self.file_names[scene_name]) for scene_name in self.file_names])


class SurfaceNet3DDataModule(pl.LightningModule):
    def __init__(self, data_config: SurfaceNet3DDataConfig):
        super().__init__()
        self.save_hyperparameters()
        self.data_config = data_config

        self.base_dataset = SceneDataset(
            camera=self.data_config.camera,
            data_dir=self.data_config.data_dir,
            scenes=self.data_config.scenes,
        )

        self.grid_dataset = VoxelGridDataset(self.base_dataset, self.data_config)

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        self.grid_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Create base dataset
        

        # Wrap with VoxelGridDataset

        # Calculate split sizes (60%, 20%, 20%)
        n_samples = len(self.grid_dataset)
        n_train = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)
        n_test = n_samples - n_train - n_val

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.grid_dataset,
            [n_train, n_val, n_test],
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
