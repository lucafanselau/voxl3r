from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, List, TypedDict
import zlib

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

    folder_name_image: str = "chunk_pairing"
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
        base_folder_name = f"num_pairs_{dc.num_pairs}_chunk_extent_{dc.chunk_extent}"

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
        return chunk_dir.exists() and len(list(chunk_dir.iterdir())) > 0

    @torch.no_grad()
    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[tuple[dict, str], None, None]:

        camera_params = base_dataset_dict["camera_params"]
        image_names_global = list(camera_params.keys())
        simplified_mesh = base_dataset_dict["score_dict"]["simplified_mesh"]
        mesh_bounds = torch.from_numpy(simplified_mesh.bounds)
        mesh_extent = torch.from_numpy(simplified_mesh.extents)
        chunk_extent = torch.from_numpy(self.data_config.chunk_extent)

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
                            "mesh_extent": mesh_extent,
                            "identifier": f"{i}_{j}_{k}",
                            "image_pairs": [],
                        }
                    )

        # chunk_considered
        if len(image_names_global) > 1400:
            logger.info(
                f"Scene {base_dataset_dict['scene_name']} has more than 1400 images"
            )
            return

        def chunk_2_bounds(chunk: dict):
            return np.array(
                [
                    chunk["chunk_center"] - chunk["chunk_extent"] / 2,
                    chunk["chunk_center"] + chunk["chunk_extent"] / 2,
                ]
            )

        for chunk in chunk_considered:
            bounds = chunk_2_bounds(chunk)
            contains = trimesh.bounds.contains(bounds, simplified_mesh.vertices)
            chunk["vertex_mask"] = torch.from_numpy(np.packbits(contains))

        score_dict = base_dataset_dict["score_dict"]

        decompressed_bytes = zlib.decompress(score_dict["image_bit_vectors_packed"])
        vertices_in_image_packed = torch.from_numpy(
            np.frombuffer(decompressed_bytes, dtype=np.uint8)
            .copy()
            .reshape(score_dict["shape_image_bit_vectors_packed"])
        )

        # intersection_vertices_global = torch.bitwise_and(
        #     vertices_in_image_packed.unsqueeze(0), vertices_in_image_packed.unsqueeze(1)
        # )
        vertices_unions_global = score_dict["vertices_unions"]
        alpha_scores_global = 4 * score_dict["alpha_scores"]

        evaluated_chunks = []

        # h, w = (
        #     camera_params[image_names[0]]["height"],
        #     camera_params[image_names[0]]["width"],
        # )

        # T_cw_batched = torch.from_numpy(np.stack([camera_params[image_name]["T_cw"] for image_name in image_names]))
        # K_batched = torch.from_numpy(np.stack([camera_params[image_name]["K"] for image_name in image_names]))

        for chunk_dict in tqdm(
            chunk_considered,
            position=1,
            desc=f"Chunking scene {base_dataset_dict['scene_name']}",
        ):
            p_c = chunk_dict["vertex_mask"]

            # OPTIMIZATION: compute the bitmask operations only for images that overlap with the chunk
            # compute only images that are visible in the chunk
            chunk_image_mask = (
                np.bitwise_count(
                    torch.bitwise_and(p_c, vertices_in_image_packed).numpy()
                ).sum(axis=-1)
                > 0
            )

            if chunk_image_mask.sum() == 0:
                # this chunk is invalid, skip it
                continue

            # build the "local" version of the variables: intersection_vertices, vertices_unions, alpha_scores, image_names

            intersection_vertices = torch.bitwise_and(
                vertices_in_image_packed[chunk_image_mask].unsqueeze(0),
                vertices_in_image_packed[chunk_image_mask].unsqueeze(1),
            )
            # intersection_vertices = intersection_vertices_global[chunk_image_mask][
            #     :, chunk_image_mask
            # ]
            vertices_unions = vertices_unions_global[chunk_image_mask][
                :, chunk_image_mask
            ]
            alpha_scores = alpha_scores_global[chunk_image_mask][:, chunk_image_mask]
            image_names = [
                name for i, name in enumerate(image_names_global) if chunk_image_mask[i]
            ]

            chunk_score = 0.0

            # center_torch = chunk_dict["chunk_center"].unsqueeze(0)
            # scene_processed.compute_visible_points_batched(T_cw_batched, K_batched, center_torch, h, w, -1, -1)

            for i in range(self.data_config.num_pairs):

                intersection_sum = np.bitwise_count(
                    torch.bitwise_and(p_c, intersection_vertices).numpy()
                ).sum(axis=-1)
                score = (alpha_scores * intersection_sum) / vertices_unions
                score[
                    range(intersection_sum.shape[0]), range(intersection_sum.shape[0])
                ] = -2

                if len(chunk_dict["image_pairs"]) > 0:
                    for image_pair in chunk_dict["image_pairs"]:
                        (idx_1, idx_2) = image_pair[3]
                        score[idx_1] = score[:, idx_1] = -2.0
                        score[idx_2] = score[:, idx_2] = -2.0

                idx_1, idx_2 = np.unravel_index(score.argmax(), score.shape)

                chunk_score += score[idx_1][idx_2]

                chunk_dict["image_pairs"].append(
                    (
                        image_names[idx_1],
                        image_names[idx_2],
                        score[idx_1][idx_2],
                        # this is only used for the score computation
                        # THIS IS A LOCAL INDEX (EG. NOT THE GLOBAL INDEX OF THE IMAGE)
                        (idx_1, idx_2),
                    )
                )

                # update p_c_(n+1) = p_c_(n) - (p_i1 *intersection+ p_i2) *intersection* rand
                # rand uint8 of size p_c.shape
                rand_mask = torch.from_numpy(
                    np.packbits(
                        torch.rand(intersection_vertices[idx_1, idx_2].shape[0] * 8)
                        < 0.5
                    )
                )
                intersection_vertices_masked = torch.bitwise_and(
                    intersection_vertices[idx_1, idx_2], rand_mask
                )
                p_c_old = p_c
                p_c = torch.bitwise_xor(p_c, intersection_vertices_masked)
                p_c = torch.bitwise_and(p_c, p_c_old)

            chunk_dict["chunk_score"] = chunk_score

            evaluated_chunks.append(chunk_dict)

        # sort by score descending order
        evaluated_chunks.sort(key=lambda x: x["chunk_score"], reverse=True)

        failed_attempts = 0
        for i, chunk_dict in enumerate(evaluated_chunks):
            identifier = f"{i}_{chunk_dict['identifier']}"
            chunk_dict["identifier"] = identifier
            chunk_dict["scene_name"] = base_dataset_dict["scene_name"]
            if chunk_dict["chunk_score"] < 1e-6:
                logger.warning(
                    f"{evaluated_chunks.__len__() - i} chunks have a score below 1e-6"
                )
                break
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

        if "vertex_mask" in data_dict:
            del data_dict["vertex_mask"]

        return data_dict


def helper_function(score_dict, data_config):

    simplified_mesh = score_dict["simplified_mesh"]
    mesh_bounds = torch.from_numpy(simplified_mesh.bounds)
    mesh_extent = torch.from_numpy(simplified_mesh.extents)
    chunk_extent = torch.from_numpy(data_config.chunk_extent)
    num_chunks = torch.ceil((mesh_extent / chunk_extent) - 0.1).int()

    return num_chunks, mesh_extent


def main():
    import visualization

    counter = 0
    while counter < 10:
        try:
            data_config = Config.load_from_files(
                [
                    "./config/data/base.yaml",
                ]
            )
            # data_config.scenes = ["2e67a32314"]
            data_config.chunk_num_workers = 1
            data_config.split = "train"

            base_dataset = scene_processed.Dataset(data_config)
            # data_config.scenes = [base_dataset.scenes[42]]
            # base_dataset = scene_processed.Dataset(data_config)
            base_dataset.load_paths()

            # list_chunks = []
            # list_extents = []
            # list_num_chunks = []
            # for scene_name in tqdm(base_dataset.scenes):
            #     data = base_dataset.get_scoring_dict(scene_name)
            #     num_chunks, mesh_extent = helper_function(data, data_config)
            #     list_chunks.append(num_chunks)
            #     list_extents.append(mesh_extent)
            #     list_num_chunks.append(num_chunks[0]*num_chunks[1]*num_chunks[2])

            data_config.skip_prepare = False

            dataset = Dataset(data_config, base_dataset)
            dataset.prepare_data()

            abc = 0
        except Exception as e:
            counter += 1
            logger.error(f"Error preparing data: {e}")

    dataset.load_paths()

    logger.info(f"Dataset length: {len(dataset)}")

    return

    chunk_dicts = sorted(list(dataset), key=lambda x: x["chunk_score"], reverse=True)
    chunk_dict = chunk_dicts[-1]
    base_data = base_dataset[0]

    visualizer_config = visualization.Config(
        log_dir=".visualization", **data_config.model_dump()
    )
    visualizer = visualization.Visualizer(visualizer_config)
    simplified_mesh = base_data["score_dict"]["simplified_mesh"]
    visualizer.add_mesh(base_data["score_dict"]["simplified_mesh"])
    images_to_visualize = [
        image_name
        for image_pairs in chunk_dict["image_pairs"]
        for image_name in image_pairs[:2]
    ]

    image_dict = {
        "cameras": [
            base_data["camera_params"][image_name] for image_name in images_to_visualize
        ],
        "images": [
            base_data["path_images"] / image_name for image_name in images_to_visualize
        ],
    }
    visualizer.add_from_image_dict(image_dict)

    vertices_idx_chunk = np.where(
        np.unpackbits(chunk_dict["vertex_mask"])[: simplified_mesh.vertices.shape[0]]
    )
    visualizer.add_points(
        torch.from_numpy(simplified_mesh.vertices[vertices_idx_chunk]), color="red"
    )
    visualizer.add_points(
        chunk_dict["chunk_center"], color="red", p_size=40, opacity=0.5
    )

    visualizer.export_html("out", timestamp=True)


if __name__ == "__main__":
    main()
