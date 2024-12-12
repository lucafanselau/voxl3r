from pathlib import Path
from typing import Optional
from einops import rearrange
import numpy as np
from jaxtyping import Float, Int
from scipy.spatial.distance import cdist
import torch
import trimesh

# from models.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from utils.transformations import invert_pose, project_image_plane


def get_images_with_3d_point(
    points, camera_params, keys=None, tolerance=0.9, max_seq_len=8
):
    """
    Detect images that contain the 3D point within a certain tolerance
    :param points: 3D points
    :param camera_params: camera parameters
    :param tolerance: tolerance
    :return: images names, pixel coordinates, camera parameters list
    """

    images_names = []
    pixel_coordinate = []
    camera_params_list = {}

    uvs_dict, _ = project_image_plane(camera_params, points)

    counter = 0

    for key in camera_params.keys() if keys is None else keys:
        c_dict = camera_params[key]

        if uvs_dict[key] is not None:
            u, v = uvs_dict[key][:, 0]
            w_min, w_max = (0.5 - tolerance / 2) * c_dict["width"], (
                0.5 + tolerance / 2
            ) * c_dict["width"]
            h_min, h_max = (0.5 - tolerance / 2) * c_dict["height"], (
                0.5 + tolerance / 2
            ) * c_dict["height"]
            if w_min <= u < w_max and h_min <= v < h_max:

                images_names.append(key)
                pixel_coordinate.append([u, v])
                camera_params_list[key] = c_dict

                counter += 1

                if counter == max_seq_len:
                    return images_names, pixel_coordinate, camera_params_list

    return images_names, pixel_coordinate, camera_params_list


def mesh_2_voxels(
    mesh, voxel_size=0.01, to_world_coordinates: Optional[np.ndarray] = None
):
    voxel_grid = mesh.voxelized(voxel_size)
    if to_world_coordinates is not None:
        voxel_grid.apply_transform(to_world_coordinates)
    occupancy_grid = voxel_grid.encoding.dense
    indices = np.indices(occupancy_grid.shape)
    origin = voxel_grid.bounds[0].reshape(3, 1, 1, 1)
    coordinate_grid = origin + (indices + 0.5) * voxel_size

    return voxel_grid, coordinate_grid, occupancy_grid


# def compute_feature_grid(occupancy_grid, data_dict, image_transform):
#     image_folder = Path(data_dict["images"][0][0]).parents[0]
#     image_dict = {
#         Path(key).name: value
#         for key, value in zip(data_dict["images"][0], data_dict["images"][1])
#     }

#     # compute the coordinates of each point in shace
#     image_name = data_dict["image_name_chunk"]
#     T_cw = image_dict[image_name]["T_cw"]
#     _, _, T_wc = invert_pose(T_cw[:3, :3], T_cw[:3, 3])
#     coordinates = compute_coordinates(
#         occupancy_grid,
#         data_dict["center"],
#         data_dict["resolution"],
#         data_dict["grid_size"][0],
#         to_world_coordinates=T_wc,
#     )
#     coordinates = torch.from_numpy(coordinates).float().to(occupancy_grid.device)

#     # transform images into space
#     chunk_data = {}
#     chunk_data["image_names"] = [
#         image_folder / image for image in image_dict.keys()
#     ]
#     chunk_data["camera_params"] = image_dict
#     images, transformations, T_cw = image_transform.forward(chunk_data)
#     feature_grid = project_voxel_grid_to_images_seperate(
#         coordinates,
#         images,
#         transformations,
#         T_cw,
#     )
#     return feature_grid


def compute_coordinates(
    grid_size: Int[np.ndarray, "3"],
    center: Float[np.ndarray, "3"],
    pitch: float,
    final_dim: int,
    to_world_coordinates: Optional[np.ndarray] = None,
):
    radius = final_dim // 2
    indices = np.indices(grid_size)
    origin = (center - np.array(3 * [radius * pitch])).reshape(
        3, 1, 1, 1
    )  # voxel_grid.bounds[0].reshape(3, 1, 1, 1)
    coordinate_grid = origin + (indices + 0.5) * pitch

    if to_world_coordinates is not None:
        coordinates = rearrange(coordinate_grid, "c x y z -> (x y z) c 1")
        # make coordinates homographic
        coordinates = np.concatenate(
            [coordinates, np.ones((coordinates.shape[0], 1, 1))], axis=1
        )
        coordinate_grid = to_world_coordinates[:3, :] @ coordinates
        coordinate_grid = rearrange(
            coordinate_grid,
            "(x y z) c 1 -> c x y z",
            x=final_dim,
            y=final_dim,
            z=final_dim,
        )

    return coordinate_grid


def mesh_2_local_voxels(
    mesh,
    center: Float[np.ndarray, "3"],
    pitch: float,
    final_dim: int,
    to_world_coordinates: Optional[np.ndarray] = None,
):
    offsetted_center = center + pitch
    radius = final_dim // 2
    voxel_grid = trimesh.voxel.creation.local_voxelize(
        mesh, offsetted_center, pitch, radius.item()
    )
    
    if voxel_grid is None:
        return None, None, np.zeros((final_dim, final_dim, final_dim))
    
    occupancy_grid = voxel_grid.encoding.dense[:-1, :-1, :-1]
    indices = np.indices(occupancy_grid.shape)
    origin = (center - np.array(3 * [radius * pitch])).reshape(
        3, 1, 1, 1
    )  # voxel_grid.bounds[0].reshape(3, 1, 1, 1)
    coordinate_grid = origin + (indices + 0.5) * pitch

    if to_world_coordinates is not None:
        coordinates = rearrange(coordinate_grid, "c x y z -> (x y z) c 1")
        # make coordinates homographic
        coordinates = np.concatenate(
            [coordinates, np.ones((coordinates.shape[0], 1, 1))], axis=1
        )
        coordinate_grid = to_world_coordinates[:3, :] @ coordinates
        coordinate_grid = rearrange(
            coordinate_grid,
            "(x y z) c 1 -> c x y z",
            x=final_dim,
            y=final_dim,
            z=final_dim,
        )
        voxel_grid.apply_transform(to_world_coordinates)

    return voxel_grid, coordinate_grid, occupancy_grid


def select_spread_out_points_with_names(
    points_dict, fixed_image_name, num_points_to_select
):

    fixed_point = points_dict[fixed_image_name]
    selected_images = [fixed_image_name]
    selected_points = [fixed_point]

    remaining_points = {k: v for k, v in points_dict.items() if k != fixed_image_name}

    remaining_images = list(remaining_points.keys())
    remaining_coords = np.array(list(remaining_points.values()))
    max_distance = 1.0

    mask = cdist(remaining_coords, np.array(selected_points)) < 1.0
    while mask.sum() < num_points_to_select:
        max_distance += 0.25
        mask = cdist(remaining_coords, np.array(selected_points)) < max_distance
        if max_distance > 10.0:
            break

    remaining_images = [s for s, valid in zip(remaining_images, mask) if valid]
    remaining_coords = remaining_coords[mask.flatten()]

    for _ in range(num_points_to_select):

        distances = cdist(remaining_coords, np.array(selected_points))
        min_distances = distances.min(axis=1)
        idx = np.argmax(min_distances)
        selected_images.append(remaining_images[idx])
        selected_points.append(remaining_coords[idx])
        remaining_coords = np.delete(remaining_coords, idx, 0)
        del remaining_images[idx]

    return selected_images, selected_points

def retrieve_images_for_chunk(camera_params_scene, image_name, max_seq_len, center, with_furthest_displacement, image_path):
    """
    Retrieve images for a chunk
    :param camera_params_scene: camera dict that is returned from SceneDataset
    :param image_name: image name which is used to create a new chunk
    :param max_seq_len: maximum number of images to retrieve
    :param center: center of the chunk in coordinate frame of the image with image_name
    :param with_furthest_displacement: sample images with furthest displacement in greedy fashion
    :return: transformation, center, camera parameters list
    """
    transformation = camera_params_scene[image_name]["T_cw"]
    _, _, back_transformation = invert_pose(
        transformation[:3, :3], transformation[:3, 3]
    )

    vec_center = np.array([*center.flatten(), 1])
    p_center = (back_transformation @ vec_center).flatten()[:3]

    image_names = sorted(list(camera_params_scene.keys()))
    idx = image_names.index(image_name)
    
    if with_furthest_displacement:
        image_names, pixel_coordinate, camera_params_list = get_images_with_3d_point(
            p_center,
            camera_params_scene,
            keys=image_names,
            tolerance=0.5,
            max_seq_len=len(image_names),
        )
        t_wc = (
            camera_params_scene[image_name]["R_cw"].T
            @ camera_params_scene[image_name]["t_cw"]
        )

        camera_centers = {image_name: t_wc}

        for key in camera_params_list.keys():
            camera_centers[key] = (
                camera_params_scene[key]["R_cw"].T @ camera_params_scene[key]["t_cw"]
            ).flatten()

        image_names, _ = select_spread_out_points_with_names(
            camera_centers, image_name, max_seq_len - 1
        )
        camera_params_list = {key: camera_params_scene[key] for key in image_names}

    else:
        image_names, pixel_coordinate, camera_params_list = get_images_with_3d_point(
            p_center,
            camera_params_scene,
            keys=image_names[idx:],
            tolerance=0.8,
            max_seq_len=max_seq_len,
        )
        
    if image_path is not None:
        image_names = [image_path / image_name for image_name in image_names]
    
    return camera_params_list, image_names

def create_chunk(
    mesh,
    image_name,
    camera_params_scene,
    center=np.array([0.0, 0.0, 1.25]),
    size=np.array([1.5, 1.0, 2.0]),
    max_seq_len=8,
    image_path=None,
    with_backtransform=True,
    with_furthest_displacement=False,
):
    transformation = camera_params_scene[image_name]["T_cw"]
    _, _, back_transformation = invert_pose(
        transformation[:3, :3], transformation[:3, 3]
    )

    vec_center = np.array([*center.flatten(), 1])
    p_center = (back_transformation @ vec_center).flatten()[:3]

    image_names = sorted(list(camera_params_scene.keys()))
    idx = image_names.index(image_name)
    if with_furthest_displacement:
        image_names, pixel_coordinate, camera_params_list = get_images_with_3d_point(
            p_center,
            camera_params_scene,
            keys=image_names,
            tolerance=0.5,
            max_seq_len=len(image_names),
        )
        t_wc = (
            camera_params_scene[image_name]["R_cw"].T
            @ camera_params_scene[image_name]["t_cw"]
        )

        camera_centers = {image_name: t_wc}

        for key in camera_params_list.keys():
            camera_centers[key] = (
                camera_params_scene[key]["R_cw"].T @ camera_params_scene[key]["t_cw"]
            ).flatten()

        image_names, _ = select_spread_out_points_with_names(
            camera_centers, image_name, max_seq_len - 1
        )
        camera_params_list = {key: camera_params_scene[key] for key in image_names}

    else:
        image_names, pixel_coordinate, camera_params_list = get_images_with_3d_point(
            p_center,
            camera_params_scene,
            keys=image_names[idx:],
            tolerance=0.8,
            max_seq_len=max_seq_len,
        )

    mesh, backtransformed = chunk_mesh(
        mesh, transformation, center, size, with_backtransform=with_backtransform
    )

    if image_path is not None:
        image_names = [image_path / image_name for image_name in image_names]

    return {
        "mesh": mesh,
        "image_names": image_names,
        "camera_params": camera_params_list,
        "p_center": p_center,
        "backtransformed": backtransformed,
    }


def chunk_mesh(
    mesh,
    transformation,
    center=np.array([0.0, 0.0, 1.25]),
    size=np.array([1.5, 1.0, 2.0]),
    with_backtransform=True,
):
    mesh.apply_transform(transformation)

    for i, offset in enumerate(size):
        plane_origin, plane_normal = np.zeros(3), np.zeros(3)
        plane_origin[i], plane_normal[i] = offset / 2, 1
        mesh = mesh.slice_plane(center - plane_origin, plane_normal)
        mesh = mesh.slice_plane(center + plane_origin, -plane_normal)

    if with_backtransform:
        _, _, back_transformation = invert_pose(
            transformation[:3, :3], transformation[:3, 3]
        )
        backprojected = mesh.copy()
        backprojected.apply_transform(back_transformation)
        return mesh, backprojected

    return mesh, None
