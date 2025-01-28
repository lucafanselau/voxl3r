from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, List, TypedDict

from loguru import logger
from shapely import area
import torch
from jaxtyping import Float
import numpy as np
from tqdm import tqdm#

from datasets import scene
from datasets.chunk.image_heuristics import calculate_norm_to_current
from datasets.scene import (
    Dataset,
    Config,
)
from utils.transformations import invert_pose_batched
from .base import ChunkBaseDataset, ChunkBaseDatasetConfig
from utils.chunking import (
    retrieve_images_for_chunk,
)
from datasets.chunk.image_heuristics import Heuristics

def get_best_idx(current, extrinsics_cw, extrinsics_wc, intrinsics, grid_config, heuristics, heuristics_dict):
    image_scores_individual = [weight * h(current, extrinsics_cw, extrinsics_wc, intrinsics, grid_config) for (h, (name, weight)) in zip(heuristics, heuristics_dict)]
    image_scores = sum(image_scores_individual)
    # disallow taking the same image twice
    image_scores[current] = -torch.inf
    best_idx = torch.argmax(image_scores)
    return best_idx, image_scores, image_scores_individual

class Config(ChunkBaseDatasetConfig, scene.Config):
    # Image Config
    seq_len: int = 4
    with_furthest_displacement: bool = False
    center_point: Float[np.ndarray, "3"] = field(
        default_factory=lambda: np.array(
            [0.0, 0.0, 1.5]
        )  # center point of chunk in camera coodinates
    )
    grid_resolution: float = 0.02
    grid_size: list[int] = field(
        default_factory=lambda: [64, 64, 64]
    )

    folder_name_image: str = "corresponding_images"
    heuristic: Optional[list[tuple[str, float]]] = field(default_factory=lambda: [("AreaUnderIntrinsics", 1.0), ("AngleHeuristics", 0.7), ("IsClose", 1.0)])
    avg_volume_per_chunk: float = 3.55

class Output(TypedDict):
    scene_name: str
    images: List[str] # Tuple[List[str], List[Dict[str, float]]]
    cameras: List[Dict[str, float]]
    center: Float[np.ndarray, "3"]
    image_name_chunk: str

class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: Dataset,
        transform: Optional[callable] = None,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.data_config = data_config
        self.transform = transform
        self.file_names = None
        self.image_cache = {}
        
        self.heuristic = None
        if self.data_config.heuristic is not None:
            assert self.data_config.avg_volume_per_chunk is not None, "avg_volume_per_chunk must be set if heuristic is used"
            self.heuristic = [Heuristics[h[0]]() for h in self.data_config.heuristic]

    def get_chunk_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        dc = self.data_config
        selection_mechanism = f"_heuristic_{self.data_config.heuristic}_avg_volume_{self.data_config.avg_volume_per_chunk}" if self.data_config.heuristic is not None else f"_furthest_{dc.with_furthest_displacement}"
        base_folder_name = f"seq_len_{dc.seq_len}{selection_mechanism}_center_{dc.center_point}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_image
            / base_folder_name
        )

        return path

    def get_chunks_of_scene(self, scene_name: str) -> List[Path]:
        chunk_dir = self.get_chunk_dir(scene_name)
        files = [s for s in chunk_dir.iterdir() if s.is_file()]
        return files
    
    def check_chunks_exists(self, scene_name: str) -> bool:
        chunk_dir = self.get_chunk_dir(scene_name)
        # here existance is enough
        return chunk_dir.exists()

    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[dict, None, None]:
        
        image_path = base_dataset_dict["path_images"]
        camera_params = base_dataset_dict["camera_params"]
        scene_name = base_dataset_dict["scene_name"]
        seq_len = self.data_config.seq_len
        image_names = list(camera_params.keys())

        
        if  self.heuristic is not None:   
            scene_volume = base_dataset_dict["bounds"][0] * base_dataset_dict["bounds"][1] * base_dataset_dict["bounds"][2]
            target_chunks = int(scene_volume / self.data_config.avg_volume_per_chunk)

            # Let's start with 8 * the target chunks
            considered_chunks = target_chunks * 8

            # let's create considered_chunks chunks and their associated scores
            chunks: list[tuple[dict, float]] = []
            step_size: int = int(len(image_names) // considered_chunks)
            
            if step_size == 0:
                step_size = 1
            
            grid_config = {
                "grid_resolution": self.data_config.grid_resolution,
                "grid_size": torch.tensor(self.data_config.grid_size),
                "center": torch.tensor(self.data_config.center_point)
            }
            extrinsics_cw = torch.from_numpy(np.array([camera_params[key]["T_cw"] for key in camera_params.keys()])).float()
            extrinsics_wc = invert_pose_batched(extrinsics_cw[:, :3, :3], extrinsics_cw[:, :3, 3]).float()
            intrinsics = torch.from_numpy(np.array([camera_params[key]["K"] for key in camera_params.keys()])).float()
            
            for i in tqdm(range((len(image_names) // step_size)), leave=False, position=1):

                chunk_score = 0
                current = [i*step_size]
    
                for i in range(self.data_config.seq_len - 1):
                    best_idx, image_scores, _ = get_best_idx(current, extrinsics_cw, extrinsics_wc, intrinsics, grid_config, heuristics=self.heuristic, heuristics_dict=self.data_config.heuristic)
                    current.append(best_idx.item())
                    chunk_score += image_scores[best_idx].item()

                # create the result dict and store it with the score
                result_dict = {
                    "scene_name": scene_name,
                    "center": self.data_config.center_point,
                    "image_name_chunk": image_names[current[0]],
                    "images": [str(image_path / image_names[i]) for i in current],
                    "cameras": [camera_params[image_names[i]] for i in current]
                }
                
                if chunk_score != -torch.inf:
                    chunks.append((result_dict, chunk_score))

            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            sorting_indices = sorted(range(len(chunks)), key=lambda k: chunks[k][1], reverse=True)
            extrinsics_wc_sorted = extrinsics_wc[sorting_indices]
            
            current = [sorting_indices[0]]
            for _ in range(target_chunks):
                score = calculate_norm_to_current(current, extrinsics_wc_sorted).min(dim=-1).values
                top_k = torch.topk(score, 5, dim=0)
                # if the top 5 are very close to each other, we can assume they are equally good and take
                # the min index (score is sorted by heuristic score)
                indices_of_interest = top_k.indices[torch.max(top_k.values) - top_k.values < 0.1]
                current.append(indices_of_interest.min().item())

            chunk_dicts = [sorted_chunks[i][0] for i in current]
            scene_score = sum([sorted_chunks[i][1] for i in current])
            
            # # now we only want the top target_chunks chunks (eg. only with maximal score)
            # # heapq is a good way to do this
            # chunks = heapq.nlargest(target_chunks, chunks, key=lambda x: x[1])
            # chunk_dicts = [chunk[0] for chunk in chunks]
            # scene_score = sum([chunk[1] for chunk in chunks])

            logger.info(f"Scene score {scene_name}: {scene_score}")

            for chunk_dict in chunk_dicts:
                yield chunk_dict, chunk_dict["image_name_chunk"]
        else:
            with_furthest_displacement = self.data_config.with_furthest_displacement
            center = self.data_config.center_point

            for i in tqdm(range((len(image_names) // seq_len)), leave=False):
                try:
                    image_name = image_names[i * seq_len]
                    camera_params_chunk, image_names_chunk = retrieve_images_for_chunk(
                        camera_params,
                        image_name,
                        seq_len,
                        center,
                        with_furthest_displacement,
                        image_path,
                    )

                    result_dict: Output = {
                        "scene_name": scene_name,
                        "center": center,
                        "image_name_chunk": image_name,
                        "images": [str(name) for name in image_names_chunk],
                        "cameras": list(camera_params_chunk.values()),
                    }

                    yield result_dict, image_name
                except Exception as e:
                    logger.error(f"[ChunkImageDataset] Error creating chunk for scene {scene_name}: {e}")
                    continue
            
    def on_after_prepare(self):
        for i in range(len(self)):
            self.get_at_idx(i)

    def get_at_idx(self, idx: int, fallback: Optional[bool] = False):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")

            return self.get_at_idx(idx - 1) if fallback else None
        if os.path.getsize(file) < 0:
            print(f"File {file} is empty. Skipping.")
            return self.get_at_idx(idx - 1) if fallback else None

        try: 
            if str(file) not in self.image_cache:
                self.image_cache[str(file)] = torch.load(file)
                
            data_dict = self.image_cache[str(file)]
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None

        return data_dict
    
if __name__ == "__main__":
    import visualization
    data_config = Config.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
    ])
    #data_config.scenes = ["2e67a32314"]
    #data_config.chunk_num_workers = 1
    
    base_dataset = scene.Dataset(data_config)
    base_dataset.load_paths() 
    
    data_config.skip_prepare = False
    data_config.force_prepare = True
        
    image_dataset = Dataset(data_config, base_dataset)
    image_dataset.prepare_data()
    
    visualizer_config = visualization.Config(log_dir=".visualization", **data_config.model_dump())
    visualizer = visualization.Visualizer(visualizer_config)
    
    for image in image_dataset:
        # just visualize the first
        image_dict = {
            "cameras": [image["cameras"][0]],
            "images": [image["images"][0]]
        }
        visualizer.add_from_image_dict(image_dict)
    
    visualizer.export_html("out", timestamp=True)

    