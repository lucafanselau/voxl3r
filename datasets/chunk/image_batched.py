from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, List, TypedDict
import zlib

from einops import rearrange
from loguru import logger
from shapely import area
import torch
from jaxtyping import Float
import numpy as np
from tqdm import tqdm
import trimesh  #

from datasets.chunk.image_heuristics import calculate_norm_to_current
from datasets import scene_processed

from utils.transformations import invert_pose, invert_pose_batched
from .base import ChunkBaseDataset, ChunkBaseDatasetConfig
from utils.chunking import (
    retrieve_images_for_chunk,
)
from datasets.chunk.image_heuristics import Heuristics


def get_best_idx(
    current,
    extrinsics_cw,
    extrinsics_wc,
    intrinsics,
    grid_config,
    heuristics,
    heuristics_dict,
):
    image_scores_individual = [
        weight * h(current, extrinsics_cw, extrinsics_wc, intrinsics, grid_config)
        for (h, (name, weight)) in zip(heuristics, heuristics_dict)
    ]
    image_scores = sum(image_scores_individual)
    # disallow taking the same image twice
    image_scores[current] = -torch.inf
    best_idx = torch.argmax(image_scores)
    return best_idx, image_scores, image_scores_individual


class Config(ChunkBaseDatasetConfig, scene_processed.Config):
    # Image Config
    num_pairs: int = 4
    with_furthest_displacement: bool = False
    chunk_extent: Float[np.ndarray, "3"] = np.array([1.5, 1.5, 1.5])

    folder_name_image: str = "corresponding_images"
    heuristic: Optional[list[tuple[str, float]]] = field(
        default_factory=lambda: [
            ("AreaUnderIntrinsics", 1.0),
            ("AngleHeuristics", 0.7),
            ("IsClose", 1.0),
        ]
    )
    avg_volume_per_chunk: float = 3.55


class Output(TypedDict):
    scene_name: str
    images: List[str]  # Tuple[List[str], List[Dict[str, float]]]
    cameras: List[Dict[str, float]]
    center: Float[np.ndarray, "3"]
    image_name_chunk: str


class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: scene_processed.Dataset,
        transform: Optional[callable] = None,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.data_config = data_config
        self.transform = transform
        self.file_names = None
        self.image_cache = {}

        self.heuristic = None
        if self.data_config.heuristic is not None:
            assert (
                self.data_config.avg_volume_per_chunk is not None
            ), "avg_volume_per_chunk must be set if heuristic is used"
            self.heuristic = [Heuristics[h[0]]() for h in self.data_config.heuristic]

    def get_chunk_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        dc = self.data_config
        selection_mechanism = (
            f"_heuristic_{self.data_config.heuristic}_avg_volume_{self.data_config.avg_volume_per_chunk}"
            if self.data_config.heuristic is not None
            else f"_furthest_{dc.with_furthest_displacement}"
        )
        base_folder_name = f"seq_len_{dc.num_pairs}{selection_mechanism}_chunk_extent_{dc.chunk_extent}"

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

    @torch.no_grad()
    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[tuple[dict, str], None, None]:

        camera_params = base_dataset_dict["camera_params"]
        image_names = list(camera_params.keys())
        simplified_mesh = base_dataset_dict["score_dict"]["simplified_mesh"]
        mesh_bounds = torch.from_numpy(simplified_mesh.bounds)
        mesh_extent = torch.from_numpy(simplified_mesh.extents)
        chunk_extent = self.data_config.chunk_extent

        num_chunks = torch.ceil((mesh_extent / chunk_extent) - 0.1).int()

        chunk_considered = []

        for i in range(num_chunks[0]):
            for j in range(num_chunks[1]):
                for k in range(num_chunks[2]):
                    chunk_center = (
                        mesh_bounds[0]
                        + torch.tensor([i, j, k]) * chunk_extent
                        + chunk_extent / 2
                    )
                    chunk_considered.append(
                        {
                            "chunk_center": chunk_center,
                            "chunk_extent": chunk_extent,
                            "identifier": f"{i}_{j}_{k}",
                        }
                    )

        def chunk_2_bounds(chunk: dict):
            return np.array(
                [
                    chunk["chunk_center"] - chunk["chunk_extent"] / 2.0,
                    chunk["chunk_center"] + chunk["chunk_extent"] / 2.0,
                ]
            )

        for chunk in chunk_considered:
            bounds = chunk_2_bounds(chunk)
            contains = trimesh.bounds.contains(bounds, simplified_mesh.vertices)
            chunk["vertex_mask"] = torch.from_numpy(np.packbits(contains))

        score_dict = base_dataset_dict["score_dict"]

        decompressed_bytes = zlib.decompress(score_dict["image_bit_vectors_packed"])
        vertices_in_image_packed = np.frombuffer(
            decompressed_bytes, dtype=np.uint8
        ).reshape(score_dict["shape_image_bit_vectors_packed"])

        intersection_vertices = np.bitwise_and(
            rearrange(vertices_in_image_packed, "I c -> 1 I c"),
            rearrange(vertices_in_image_packed, "J c -> J 1 c"),
        )
        vertices_unions = score_dict["vertices_unions"]
        alpha_scores = 4 * score_dict["alpha_scores"]

        # (P_1 schnitt P_2) schnitt P_c  == (P_1 schnitt P_c) schnitt (P_2 schnitt P_c)

        step = 8

        for start in tqdm(range(0, len(chunk_considered), step)):
            end = min(start + step, len(chunk_considered))

            p_c = torch.stack(
                [chunk["vertex_mask"] for chunk in chunk_considered[start:end]]
            )

            # also initialize image_pairs for each chunk
            for i in range(start, end):
                chunk_considered[i]["image_pairs"] = []
                chunk_considered[i]["chunk_score"] = 0.0

            for i in range(self.data_config.num_pairs):

                chunked_pairwise_intersections = np.bitwise_and(
                    rearrange(p_c, "N c -> N 1 1 c"),
                    rearrange(intersection_vertices, "I1 I2 c -> 1 I1 I2 c"),
                )

                # bitcount chunked pairwise intersections
                summed_intersections = np.bitwise_count(
                    chunked_pairwise_intersections.numpy()
                ).sum(axis=-1)

                # set diagonal to -2.0
                summed_intersections[
                    :,
                    range(summed_intersections.shape[0]),
                    range(summed_intersections.shape[0]),
                ] = -2.0

                batched_scores = (alpha_scores * summed_intersections) / vertices_unions

                for offset, score in enumerate(batched_scores):
                    i = start + offset
                    idx_1, idx_2 = np.unravel_index(score.argmax(), score.shape)
                    chunk_considered[i]["chunk_score"] += score[idx_1][idx_2]

                    chunk_considered[i]["image_pairs"].append(
                        (image_names[idx_1], image_names[idx_2], score[idx_1][idx_2])
                    )

                    # update p_c_(n+1) = p_c_(n) - (p_i1 *intersection+ p_i2)
                    p_c_old = p_c[offset]
                    p_c[offset] = np.bitwise_xor(
                        p_c_old, intersection_vertices[idx_1, idx_2]
                    )
                    p_c[offset] = np.bitwise_and(p_c[offset], p_c_old)

        # sort by score descending order
        chunk_considered.sort(key=lambda x: x["chunk_score"], reverse=True)

        for i, chunk_dict in enumerate(chunk_considered):
            identifier = f"{i}_{chunk_dict['identifier']}"
            chunk_dict["identifier"] = identifier
            yield chunk_dict, identifier

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
                self.image_cache[str(file)] = torch.load(file, weights_only=False)

            data_dict = self.image_cache[str(file)]
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None

        return data_dict


if __name__ == "__main__":
    import visualization

    data_config = Config.load_from_files(
        [
            "./config/data/base.yaml",
        ]
    )
    # data_config.scenes = ["2e67a32314"]
    data_config.chunk_num_workers = 1
    data_config.split = "train"

    base_dataset = scene_processed.Dataset(data_config)
    base_dataset.load_paths()

    data_config.skip_prepare = False
    image_dataset = Dataset(data_config, base_dataset)
    image_dataset.prepare_data()

    from trimesh.ray.ray_pyembree import RayMeshIntersector

    idx = 0
    image_data = image_dataset[0]
    scene_name = image_data["scene_name"]
    scene_id = base_dataset.get_index_from_scene(scene_name)
    mesh = base_dataset.get_mesh(scene_id)
    rayMeshIntersector = RayMeshIntersector(mesh, scale_to_box=False)

    image_idx = 0
    camera_params = image_data["cameras"][0]

    R_cw, t_cw = camera_params["T_cw"][:3, :3], camera_params["T_cw"][:3, 3]
    R_wc, t_wc, T_wc = invert_pose(R_cw, t_cw)

    K_inv = np.linalg.inv(camera_params["K"])

    x, y = np.arange(camera_params["width"]), np.arange(camera_params["height"])
    XX, YY = np.meshgrid(x, y)
    pixels = np.stack([XX, YY], axis=-1).reshape(-1, 2)
    ones = np.ones((pixels.shape[0], 1))
    pixels_hom = np.hstack((pixels, ones))

    pts_grid_camera = K_inv @ pixels_hom.T
    pts_grid = R_wc @ pts_grid_camera
    t_wc_repeated = np.repeat(t_wc, pts_grid.shape[1], axis=1)

    result = rayMeshIntersector.intersects_location(
        t_wc_repeated.T, pts_grid.T, multiple_hits=False
    )
    pts_intesection, ray_index = result[0], result[1]

    visualizer_config = visualization.Config(
        log_dir=".visualization", **data_config.model_dump()
    )
    visualizer = visualization.Visualizer(visualizer_config)

    visualizer.add_scene(scene_name)
    visualizer.add_from_image_dict(image_data)
    visualizer.add_points(torch.from_numpy(pts_intesection))

    visualizer.export_html("out", timestamp=True)
