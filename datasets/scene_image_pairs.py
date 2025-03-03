from multiprocessing import Pool
from pathlib import Path
import time
from typing import Optional
from typing import List
import zlib
from loguru import logger
from matplotlib import spines
import pandas as pd
import tqdm
from utils.config import BaseConfig
import torch
from datasets import scene
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector

import pyvista as pv
import trimesh
from torch import nn

from utils.transformations import invert_pose, project_image_plane


class Config(scene.Config):
    storage_score_dicts: str = "score_dicts"
    force_prepare_scoring_dicts: bool = False


def get_valid_face_ids(rayMeshIntersector, t_wc, pts):
    pts_grid = pts - t_wc
    t_wc_repeated = np.repeat(np.expand_dims(t_wc, axis=0), pts_grid.shape[0], axis=0)

    idx_faces, _ = rayMeshIntersector.intersects_id(
        t_wc_repeated, pts_grid, multiple_hits=False, return_locations=False
    )

    return np.unique(idx_faces)


import torch


@torch.no_grad()
def compute_visible_points_batched(
    T_cw_batched,
    K_batched,
    pts_of_interest,
    height,
    width,
    pt_batch_size=10000,
    cam_batch_size=10,
):
    """
    Computes the visible point indices for a set of cameras in batches.

    Args:
        T_cw_batched (torch.Tensor): [N_cam, 4, 4] camera extrinsics (world -> camera).
        K_batched (torch.Tensor): [N_cam, 3, 3] camera intrinsics.
        pts_of_interest (torch.Tensor): [N_pts, 3] points in world coordinates.
        height (int): image height.
        width (int): image width.
        pt_batch_size (int): number of points to process at once.
        cam_batch_size (int): number of cameras to process at once.

    Returns:
        visible_points_list (list): A list of length N_cam, where each element is a 1D tensor
                                    containing the indices of pts_of_interest visible in that camera.
    """
    num_cameras = T_cw_batched.shape[0]
    num_points = pts_of_interest.shape[0]

    # Prepare a list to store visible indices for each camera.
    visible_points_list = [[] for _ in range(num_cameras)]

    # Loop over cameras in batches.
    for cam_start in range(0, num_cameras, cam_batch_size):
        cam_end = min(cam_start + cam_batch_size, num_cameras)
        T_cam_batch = T_cw_batched[cam_start:cam_end]  # Shape: [B, 4, 4]
        K_cam_batch = K_batched[cam_start:cam_end]  # Shape: [B, 3, 3]
        batch_size = T_cam_batch.shape[0]

        # Process points in batches to keep memory usage fixed.
        for pt_start in range(0, num_points, pt_batch_size):
            pt_end = min(pt_start + pt_batch_size, num_points)
            pts_batch = pts_of_interest[pt_start:pt_end]  # Shape: [P, 3]

            # Convert to homogeneous coordinates: [P, 4]
            ones = torch.ones(
                pts_batch.shape[0], 1, dtype=pts_batch.dtype, device=pts_batch.device
            )
            pts_batch_h = torch.cat([pts_batch, ones], dim=1)  # [P, 4]

            # Transform points from world to camera coordinates for all cameras in the batch.
            # Using broadcasting: [B, 4, 4] @ [4, P] -> [B, 4, P], then transpose to [B, P, 4].
            pts_cam = (T_cam_batch @ pts_batch_h.T).transpose(1, 2)  # [B, P, 4]

            # Convert to non-homogeneous coordinates.
            pts_cam_xyz = pts_cam[..., :3] / pts_cam[..., 3:4]  # [B, P, 3]

            # Check if points are in front of each camera (positive z).
            in_front = pts_cam_xyz[..., 2] > 0  # [B, P]

            # Project the 3D points using the intrinsic matrices.
            # Transpose pts_cam_xyz to [B, 3, P] for matrix multiplication.
            pts_proj = K_cam_batch @ pts_cam_xyz.transpose(1, 2)  # [B, 3, P]
            pts_proj = (
                pts_proj / pts_proj[:, 2:3, :]
            )  # Normalize so that third coordinate is 1

            # Get pixel coordinates.
            u = pts_proj[:, 0, :]  # [B, P]
            v = pts_proj[:, 1, :]  # [B, P]

            # Check if projected points lie within image boundaries.
            within_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)  # [B, P]

            # Valid if in front and within the image.
            valid = in_front & within_bounds  # [B, P]

            # For each camera in the current batch, accumulate visible point indices.
            for i in range(batch_size):
                valid_mask = valid[i]  # [P]
                if valid_mask.any():
                    # Convert local indices in this point batch to global indices.
                    batch_indices = torch.arange(
                        pt_start, pt_end, device=pts_of_interest.device
                    )
                    visible_indices = batch_indices[valid_mask]
                    visible_points_list[cam_start + i].append(visible_indices.cpu())

        # Combine the chunks for each camera in the current batch.
        for i in range(cam_start, cam_end):
            if visible_points_list[i]:
                visible_points_list[i] = torch.cat(visible_points_list[i]).numpy()
            else:
                visible_points_list[i] = torch.empty(
                    0, dtype=torch.long, device="cpu"
                ).numpy()

    return visible_points_list


def process_camera(args):
    (
        i,
        images_to_check,
        visible_points_indices,
        pts_of_interest_numpy,
        parents_item,
        rayMeshIntersector,
        mesh,
    ) = args

    image_name = images_to_check[i]
    pts = pts_of_interest_numpy[visible_points_indices[i]]
    cam_params = parents_item["camera_params"][image_name]
    t_wc = cam_params["T_wc"][:3, 3]
    faces_id = get_valid_face_ids(rayMeshIntersector, t_wc, pts)
    vertices_id = np.unique(
        trimesh.geometry.faces_to_edges(mesh.faces[faces_id])
    ).tolist()
    return image_name, set(vertices_id)


class Dataset(scene.Dataset):
    def __init__(
        self,
        data_config: Config,
    ):
        super().__init__(data_config)
        self.file_names_score_dict = {}

    def get_file_name_scoring_dict(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        return f"{scene_name}.pt"

    def get_score_dict_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        path = (
            Path(self.data_config.data_dir)
            / self.data_config.storage_score_dicts
            / scene_name
        )
        return path

    def prepare_scene(self, scene_name: str):

        super(Dataset, self).prepare_scene(scene_name)
        data_dir = self.get_score_dict_dir(scene_name)

        if (
            data_dir / self.get_file_name_scoring_dict(scene_name)
        ).exists() and not self.data_config.force_prepare_scoring_dicts:
            return

        idx = self.get_index_from_scene(scene_name)
        score_dict = self.calculate_score_dict(
            parents_item=super(Dataset, self).__getitem__(idx)
        )

        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        torch.save(
            score_dict, data_dir / self.get_file_name_voxelized_scene(scene_name)
        )

    def get_scoring_dict(self, scene_name: str):
        data_dir = self.get_score_dict_dir(scene_name)
        return torch.load(
            data_dir / self.get_file_name_voxelized_scene(scene_name),
            weights_only=False,
        )

    def calculate_score_dict(self, parents_item):
        images_to_check = list(parents_item["camera_params"].keys())

        h, w = (
            parents_item["camera_params"][images_to_check[0]]["height"],
            parents_item["camera_params"][images_to_check[0]]["width"],
        )
        device = "cuda"
        T_cw_batched = torch.from_numpy(
            np.stack(
                [
                    parents_item["camera_params"][image_name]["T_cw"]
                    for image_name in images_to_check
                ]
            )
        ).to(device)
        K_batched = torch.from_numpy(
            np.stack(
                [
                    parents_item["camera_params"][image_name]["K"]
                    for image_name in images_to_check
                ]
            )
        ).to(device)

        mesh = parents_item["mesh"].simplify_quadric_decimation(0.95)
        pts_of_interest_numpy = mesh.triangles_center.copy()
        pts_of_interest = torch.from_numpy(pts_of_interest_numpy).to(device)
        pts_of_interest_h = torch.cat(
            [
                pts_of_interest,
                torch.ones(pts_of_interest.shape[0], 1, dtype=pts_of_interest.dtype).to(
                    device
                ),
            ],
            dim=1,
        )

        visible_points_indices = compute_visible_points_batched(
            T_cw_batched,
            K_batched,
            pts_of_interest,
            h,
            w,
            pt_batch_size=2**18,
            cam_batch_size=128,
        )

        rayMeshIntersector = RayMeshIntersector(mesh)

        # args_list = [
        #     (i, images_to_check, visible_points_indices, pts_of_interest_numpy,
        #     parents_item, rayMeshIntersector, mesh)
        #     for i in range(len(images_to_check))
        # ]

        # with Pool(2) as pool:
        #     results = list(tqdm.tqdm(pool.imap(process_camera, args_list), total=len(args_list)))
        # map_image_name_to_vertices_id = {img: vertices for img, vertices in results}

        map_image_name_to_vertices_id = {}
        # for i in tqdm.tqdm(range(len(images_to_check))):
        for i in range(len(images_to_check)):
            image_name = images_to_check[i]
            pts = pts_of_interest_numpy[visible_points_indices[i]]
            faces_id = get_valid_face_ids(
                rayMeshIntersector,
                parents_item["camera_params"][image_name]["T_wc"][:3, 3],
                pts,
            )
            # TODO: check simplifications
            vertices_id = np.unique(
                trimesh.geometry.faces_to_edges(mesh.faces[faces_id])
            ).tolist()
            map_image_name_to_vertices_id[image_name] = set(vertices_id)

        camera_z_axis = np.stack(
            [
                parents_item["camera_params"][image_name]["T_wc"][:3, 3]
                for image_name in images_to_check
            ]
        )
        norms = np.linalg.norm(camera_z_axis, axis=1, keepdims=True)
        dot_matrix = camera_z_axis @ camera_z_axis.T
        norm_matrix = norms @ norms.T
        cosine_similarity_matrix = dot_matrix / norm_matrix
        alpha_scores = cosine_similarity_matrix * (1 - cosine_similarity_matrix)

        vertices_intersection = np.zeros((len(images_to_check), len(images_to_check)))
        vertices_unions = np.zeros((len(images_to_check), len(images_to_check)))

        # Precompute boolean arrays for each image
        image_bit_vectors = np.zeros(
            (len(images_to_check), (mesh.vertices).__len__()), dtype=bool
        )

        for i, img in enumerate(images_to_check):
            image_bit_vectors[i, list(map_image_name_to_vertices_id[img])] = True

        image_bit_vectors_packed = np.packbits(image_bit_vectors, axis=1)
        image_bit_vectors_packed_compressed = zlib.compress(
            image_bit_vectors_packed.tobytes()
        )

        vertices_intersection = image_bit_vectors @ image_bit_vectors.T

        lengths = np.array(
            [len(map_image_name_to_vertices_id[img]) for img in images_to_check]
        )
        lengths_matrix = (np.tile(lengths, (len(images_to_check), 1))) + (
            np.tile(lengths, (len(images_to_check), 1))
        ).T
        vertices_unions = lengths_matrix - vertices_intersection

        score = alpha_scores * (vertices_intersection / vertices_unions)

        return {
            "alpha_scores": alpha_scores,
            "vertices_intersection": vertices_intersection,
            "vertices_unions": vertices_unions,
            "image_bit_vectors_packed": image_bit_vectors_packed_compressed,
            "shape_image_bit_vectors_packed": image_bit_vectors_packed.shape,
            "shape_image_bit_vectors": image_bit_vectors.shape,
            "score": score,
            "simplified_mesh": mesh,
            ## chunk specific stuff
            # (N_chunks, n_vertices)
        }

    def __getitem__(self, idx: int):
        parents_item = super(Dataset, self).__getitem__(idx)

        return {
            **parents_item,
            "score_dict": self.get_scoring_dict(parents_item["scene_name"]),
        }


if __name__ == "__main__":
    import visualization
    from datasets.chunk import image

    data_config = Config.load_from_files(
        [
            "./config/data/base.yaml",
        ]
    )
    data_config.split = "train"
    dataset = Dataset(data_config)
    # dataset.prepare_data()

    data = dataset[0]

    score_dict = data["score_dict"]
    simplified_mesh = score_dict["simplified_mesh"]

    decompressed_bytes = zlib.decompress(score_dict["image_bit_vectors_packed"])
    image_bit_vectors_packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    image_bit_vectors_packed = image_bit_vectors_packed.reshape(
        score_dict["shape_image_bit_vectors_packed"]
    )
    image_vertices_id = np.unpackbits(image_bit_vectors_packed, axis=1)[
        :, : simplified_mesh.vertices.__len__()
    ]

    score = score_dict["score"]
    i_max, j_max = np.unravel_index(score.argmax(), score.shape)

    vertices_i = simplified_mesh.vertices[np.where(image_vertices_id[i_max])]
    vertices_j = simplified_mesh.vertices[np.where(image_vertices_id[j_max])]

    visualizer_config = visualization.Config(
        log_dir=".visualization", **data_config.model_dump()
    )
    visualizer = visualization.Visualizer(visualizer_config)

    visualizer.add_mesh(simplified_mesh)
    images_to_check = list(data["camera_params"].keys())

    # image_dict = {
    #     "cameras" : [data["camera_params"][]],
    #     "images" : [data["path_images"] / image_names[idx_of_interest]]
    # }
    # visualizer.add_from_image_dict(data["camera_params"][image_names[idx_of_interest]])
    visualizer.add_points(torch.from_numpy(vertices_i), color="red")
    visualizer.add_points(torch.from_numpy(vertices_j), color="blue")

    visualizer.export_html("out", timestamp=True)
