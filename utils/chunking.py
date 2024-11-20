from typing import Optional
from einops import rearrange
import numpy as np
from jaxtyping import Float
import trimesh

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


def mesh_2_voxels(mesh, voxel_size=0.01, to_world_coordinates: Optional[np.ndarray] = None):
    voxel_grid = mesh.voxelized(voxel_size)
    if to_world_coordinates is not None:
        voxel_grid.apply_transform(to_world_coordinates)
    occupancy_grid = voxel_grid.encoding.dense
    indices = np.indices(occupancy_grid.shape)
    origin = voxel_grid.bounds[0].reshape(3, 1, 1, 1)
    coordinate_grid = origin + (indices + 0.5) * voxel_size

    return voxel_grid, coordinate_grid, occupancy_grid
    
    
def mesh_2_local_voxels(mesh,  center: Float[np.ndarray, "3"], pitch: float, final_dim: int, to_world_coordinates: Optional[np.ndarray] = None):
    offsetted_center = center + pitch
    radius = final_dim // 2
    voxel_grid = trimesh.voxel.creation.local_voxelize(mesh, offsetted_center, pitch, radius.item())
    occupancy_grid = voxel_grid.encoding.dense[:-1, :-1, :-1]
    indices = np.indices(occupancy_grid.shape)
    origin = (center - np.array(3*[radius*pitch])).reshape(3, 1, 1, 1)#voxel_grid.bounds[0].reshape(3, 1, 1, 1)
    coordinate_grid = (origin + (indices + 0.5) * pitch)

    if to_world_coordinates is not None:
        coordinates = rearrange(coordinate_grid, "c x y z -> (x y z) c 1")
        # make coordinates homographic
        coordinates = np.concatenate([coordinates, np.ones((coordinates.shape[0], 1, 1))], axis=1)
        coordinate_grid = to_world_coordinates[:3, :] @ coordinates
        coordinate_grid = rearrange(coordinate_grid, "(x y z) c 1 -> c x y z", x=final_dim, y=final_dim, z=final_dim)

    return voxel_grid, coordinate_grid, occupancy_grid

def create_chunk(
    mesh,
    image_name,
    camera_params_scene,
    center=np.array([0.0, 0.0, 1.25]),
    size=np.array([1.5, 1.0, 2.0]),
    max_seq_len=8,
    image_path=None,
    with_backtransform=True,
):
    transformation = camera_params_scene[image_name]["T_cw"]
    _, _, back_transformation = invert_pose(
        transformation[:3, :3], transformation[:3, 3]
    )

    vec_center = np.array([*center.flatten(), 1])
    p_center = (back_transformation @ vec_center).flatten()[:3]

    image_names = sorted(list(camera_params_scene.keys()))
    idx = image_names.index(image_name)

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
