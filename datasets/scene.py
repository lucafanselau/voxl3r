from multiprocessing import Pool
from pathlib import Path
import time
from typing import Optional
from typing import List
from loguru import logger
from matplotlib import spines
import pandas as pd
import tqdm
from utils.config import BaseConfig
import torch
from torch.utils.data import Dataset
import numpy as np

import pyvista as pv
import trimesh
from torch import nn

from common.scene_release import ScannetppScene_Release
from iphone.prepare_iphone_data import (
    extract_depth,
    extract_masks,
    extract_rgb,
)
from utils.data_parsing import (
    get_camera_params,
)

class Config(BaseConfig):
    data_dir: str = "datasets/scannetpp/"
    camera: str = "dslr"
    scenes: Optional[List[str]] = None
    
    # voxelization parameters
    storage_preprocessing_voxelized_scenes: str = "preprocessed_voxel_grids"
    num_workers_voxelization: int = 2
    force_prepare_voxelize: bool = False
    scene_resolution: float = 0.01
    return_voxelized: bool = True
    split: Optional[str] = None


class Dataset(Dataset):
    def __init__(
        self,
        data_config: Config,
    ):
        self.data_config = data_config
        self.camera = data_config.camera
        self.data_dir = Path(data_config.data_dir) / "data"
        
        if data_config.split is None:
             self.scenes = (
            [x.name for x in self.data_dir.glob("*") if x.is_dir()]
            if data_config.scenes is None
            else data_config.scenes
            )
        else:
            if data_config.split == "test":
                split_scenes_path = Path(data_config.data_dir) / "splits" / f"sem_{data_config.split}.txt"
            elif data_config.split in ["val"]:
                split_scenes_path = Path(data_config.data_dir) / "splits" / f"nvs_sem_{data_config.split}.txt"
            elif data_config.split in ["train"]:
                split_scenes_path = [Path(data_config.data_dir) / "splits" / f"nvs_sem_{data_config.split}.txt"]
                split_scenes_path.append(Path(data_config.data_dir) / "splits" / f"sem_test.txt")
            else:
                raise ValueError(f"Split {data_config.split} not supported")

            if isinstance(split_scenes_path, list):
                split_scenes = []
                for path in split_scenes_path:
                    with open(path, "r") as f:
                        split_scenes += f.read().splitlines()
            else:
                with open(split_scenes_path, "r") as f:
                    split_scenes = f.read().splitlines()
            
            if data_config.scenes is not None:
                not_in_split = set(data_config.scenes) - set(split_scenes)
                if len(not_in_split) > 0:
                    logger.warning(f"Split {data_config.split} does not contain the scenes: {not_in_split}")

                self.scenes = list(set.intersection(*[set(ids) for ids in [split_scenes, data_config.scenes]]))
            else:
                self.scenes = split_scenes
                        
    def get_saving_path(self, scene_name: str) -> Path:
        return (
            Path(self.data_config.data_dir)
            / self.data_config.storage_preprocessing_voxelized_scenes
            / scene_name
            / self.data_config.camera
        )
        
    def get_voxelized_scene_dir(self, scene_name: str, resolution: float) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        base_folder_name = f"grid_res_{resolution}"
        path = (
            Path(self.data_config.data_dir)
            / self.data_config.storage_preprocessing_voxelized_scenes
            / scene_name
            / base_folder_name
        )
        return path 
    
    def load_paths(self):
        self.file_names = {}

        for scene_name in self.scenes:
            data_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
            if data_dir.exists():
                self.file_names[scene_name] = [
                    s for s in data_dir.iterdir() if s.is_file()
                ]

        
    def prepare_data(self):
        
        if self.data_config.num_workers_voxelization > 1:
            with Pool(self.data_config.num_workers_voxelization) as p:
                with tqdm.tqdm(total=len(self.scenes), position=0, leave=True) as pbar:
                    for _ in p.imap_unordered(self.prepare_scene, self.scenes):
                        pbar.update()
        else:
            for scene_name in tqdm.tqdm(self.scenes, leave=False):
                self.prepare_scene(scene_name)
        self.load_paths()
        self.prepared = True
        
    def check_voxelized_scene_exists(self, scene_name: str) -> bool:
        voxelized_scene_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
        return voxelized_scene_dir.exists()

    def get_file_name_voxelized_scene(self, scene_name: str):
        return f"{scene_name}.pt"
        
    def prepare_scene(self, scene_name: str):

        data_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)

        if self.check_voxelized_scene_exists(scene_name) and not self.data_config.force_prepare_voxelize:
            # we need to check if all of the chunks for this scene are present
            logger.trace(f"Voxelized scenes for scene {scene_name} already exist. Skipping.")
            return
        
        mesh_path = (
            self.data_dir
            / scene_name
            / "scans"
            / "mesh_aligned_0.05.ply"
        )
        if not mesh_path.exists():
            print(f"Mesh not found for scene {scene_name}. Skipping.")
            return


        if (data_dir / self.get_file_name_voxelized_scene(scene_name)).exists():
            return
        
        voxelized_scene = self.voxelize_scene(scene_name, resolution=self.data_config.scene_resolution)
        
        if not data_dir.exists():
            data_dir.mkdir(parents=True)


        torch.save(voxelized_scene, data_dir / self.get_file_name_voxelized_scene(scene_name))
        
    def voxelize_scene(self, scene_name, resolution=0.01):
        idx = self.get_index_from_scene(scene_name)
        voxel_grid = trimesh.voxel.creation.voxelize(self.get_mesh(idx), resolution)
        return voxel_grid

    def __len__(self):
        return len(self.scenes)

    def extract_iphone(self, idx):
        scene = ScannetppScene_Release(self.scenes[idx], data_root=self.data_dir)
        extract_rgb(scene)
        extract_masks(scene)
        extract_depth(scene)

    def get_index_from_scene(self, scene_name):
        if isinstance(scene_name, list):
            return [self.scenes.index(name) for name in scene_name]
        return self.scenes.index(scene_name)
    
    def get_voxelized_scene(self, scene_name):
        voxelized_scene_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
        # maybe make this faster by caching loaded scenes
        return torch.load(voxelized_scene_dir / self.get_file_name_voxelized_scene(scene_name), weights_only=False)
    
    def get_mesh(self, idx):
        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        return trimesh.load(mesh_path)

    def __getitem__(self, idx):
        scene_path = self.data_dir / self.scenes[idx]
        camera_path = scene_path / self.camera

        if self.camera == "iphone" and not (
            (camera_path / "rgb").exists()
            and (camera_path / "rgb_masks").exists()
            and (camera_path / "depth").exists()
        ):
            self.extract_iphone(idx)

        if self.camera == "dslr" and not (
            (camera_path / "undistorted_images").exists()
        ):
            raise ValueError("Please run the undistortion script for this scene first")

        mesh = self.get_mesh(idx)

        images_with_params = get_camera_params(scene_path, self.camera, None, 0)

        image_dir = "rgb" if self.camera == "iphone" else "undistorted_images"

        bounds = mesh.extents

        shared_return = {
            "scene_name": self.scenes[idx],
            "mesh": mesh,
            "path_images": self.data_dir / self.scenes[idx] / self.camera / image_dir,
            "camera_params": images_with_params,
            "bounds": bounds
        }

        if self.data_config.return_voxelized:
            if not self.check_voxelized_scene_exists(self.scenes[idx]):
                print("Voxelizing scene does not exist yet and is created on the fly!")
                self.prepare_scene(self.scenes[idx])
            return {
                **shared_return,
                "voxelized_scene": self.get_voxelized_scene_dir(self.scenes[idx], resolution=self.data_config.scene_resolution),
            }
        
        return shared_return
    
    def find_total_volume(self, target_chunks=10_000):
        """
        Output for whole dataset:
        Total volume: 35323.74724015739
        Total scenes: 323
        Average volume: 109.36144656395477
        Average volume per chunk: 3.5323747240157393 with target chunks: 10000
        """
        sum = 0
        valid_scenes = 0
        for i in tqdm.tqdm(list(range(self.__len__()))):
            # weird way to handle missing scenes
            try:
                item = self.__getitem__(i)
                b = item["bounds"]
                sum += b[0] * b[1] * b[2]
                valid_scenes += 1
            except:
                pass

        logger.info(f"Total volume: {sum}")
        logger.info(f"Total scenes: {valid_scenes}")

        logger.info(f"Average volume: {sum / valid_scenes}")
        # calculate the average volume that is required to get target_chunks chunks
        avg_volume_per_chunk = sum / target_chunks
        logger.info(f"Average volume per chunk: {avg_volume_per_chunk} with target chunks: {target_chunks}")


        return sum
            

if __name__ == "__main__":
    data_config = Config.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
    ])
    
    #data_config.split = "train"
        
    base_dataset = Dataset(data_config)
    base_dataset.prepare_data()