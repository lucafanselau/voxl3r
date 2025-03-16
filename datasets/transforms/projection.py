from typing import Optional, Tuple
from einops import rearrange
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import lightning as pl
from torch import nn
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from positional_encodings.torch_encodings import PositionalEncoding3D

from utils.chunking import create_chunk, mesh_2_voxels

# Singleton, so that we can hook into the caching mechanism
pe = None  # PositionalEncoding3D(channels).to(grid.device)


def get_3d_pe(
    grid: Float[Tensor, "3 X Y Z"], channels: int
) -> Float[Tensor, "X*Y*Z C"]:
    """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
    :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
    """
    global pe
    if pe is None:
        pe = PositionalEncoding3D(channels).to(grid.device)

    grid = rearrange(grid, "C X Y Z -> 1 X Y Z C")

    forward = pe(grid)

    return rearrange(forward, "1 X Y Z C -> C X Y Z")


@jaxtyped(typechecker=beartype)
def project_voxel_grid_to_images_seperate(
    voxel_grid: Float[Tensor, "3 X Y Z"],
    images: Float[Tensor, "I F H W"],
    transformations: Float[Tensor, "I 3 4"],
    T_cw: Float[Tensor, "I 4 4"],
    grid_sampling_mode: Optional[str] = "bilinear",
) -> Tuple[Float[Tensor, "I F X Y Z"], Float[Tensor, "I 1 X Y Z"], Float[Tensor, "I 1 X Y Z"], Float[Tensor, "I 3 X Y Z"]]:
    """
    I = Images
    F = I * (3 + add_projected_depth + add_validity_indicator + add_viewing_directing * 3)
    images: expected to be normalized [0, 1] float images C, H, W

    returns rgb_features, projected_depth, validity_indicator, viewing_direction

    """
    _3, X, Y, Z = voxel_grid.shape

    # reshape to points
    points = rearrange(voxel_grid, "C X Y Z -> (X Y Z) C")

    # reshape points [num_points, 3] so that we have [num_points, num_images, 3]
    num_points = points.shape[0]
    num_images, _3, height, width = images.shape

    points = points.reshape(num_points, 1, 3).repeat(1, num_images, 1)

    R_cw = T_cw[:, :3, :3]
    t_cw = T_cw[:, :3, 3]
    R_wc = torch.transpose(R_cw, 1, 2)
    t_wc = -torch.matmul(R_wc, t_cw.unsqueeze(-1)).squeeze(-1)
    viewing_direction = points - t_wc
    viewing_direction = viewing_direction / torch.linalg.norm(
        viewing_direction, dim=2
    ).unsqueeze(-1)

    # convert to homographic coordinates
    points = torch.cat(
        [points, torch.full((num_points, num_images, 1), 1).to(points)], dim=-1
    )
    # and convert it to "matrix" (4x1) (num_points, num_images, 4, 1)
    points = rearrange(points, "n i d -> n i d 1")

    # perform the matmul with broadcasting (num_points, num_images, 4, 4) x (num_points, num_images, 4, 1) -> (num_points, num_images, 3, 1)
    transformed_points = torch.matmul(transformations, points)

    # remove last dimension (num_points, num_images, 3)
    transformed_points = rearrange(transformed_points, "n i d 1 -> n i d")

    transformed_points_without_K = torch.matmul(T_cw[:, :3, :], points)
    transformed_points_without_K = rearrange(
        transformed_points_without_K, "n i d 1 -> n i d"
    )
    projected_depth = transformed_points_without_K[:, :, 2].unsqueeze(-1)

    # perform perspective division
    transformed_points = transformed_points[:, :, :3] / transformed_points[
        :, :, 2
    ].unsqueeze(-1)
    # and drop the last dimension
    transformed_points = transformed_points[:, :, :2]

    # now we need to mask by the bounding box of the image and replace with a constant fill (-inf)
    fill_value = 0.0
    #fill_value = 0
    mask = (
        (transformed_points[:, :, 0] >= 0)
        & (transformed_points[:, :, 0] < width)
        & (transformed_points[:, :, 1] >= 0)
        & (transformed_points[:, :, 1] < height)
    )
    full_mask = mask.reshape(-1, num_images, 1).repeat(1, 1, 2)
    valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

    # now sample the image at the valid pixels
    # the way this is done currently all of the "invalid" uvs will be 0, but we only use the valid ones later so its fine
    grid = rearrange(valid_pixels, "p n two -> n 1 p two", two=2)
    # normalize in last dimension to range [-1, 1]
    grid = (grid / torch.tensor([width, height]).to(grid)) * 2 - 1

    sampled = torch.nn.functional.grid_sample(images, grid, align_corners=True, mode=grid_sampling_mode)
    rgb_features = rearrange(
        sampled, "images channels 1 points -> points images channels"
    )

    rgb_features[~mask] = fill_value
    validity_indicator = mask.float().unsqueeze(-1)

    rgb_features = rearrange(
        rgb_features, "points images channels -> points images channels"
    )
    
    rgb_features = rearrange(rgb_features, "(X Y Z) images channels -> images channels X Y Z", X=X, Y=Y, Z=Z)

    projected_depth = rearrange(projected_depth, "points images 1 -> points images")
    projected_depth = rearrange(projected_depth, "(X Y Z) F -> F 1 X Y Z", X=X, Y=Y, Z=Z)

    validity_indicator = rearrange(
        validity_indicator, "points images 1 -> points images"
    )
    validity_indicator = rearrange(
        validity_indicator, "(X Y Z) F -> F 1 X Y Z", X=X, Y=Y, Z=Z
    )

    viewing_direction = rearrange(
        viewing_direction, "points images channels -> points images channels"
    )
    viewing_direction = rearrange(
        viewing_direction, "(X Y Z) images channels -> images channels X Y Z", X=X, Y=Y, Z=Z
    )

    return rgb_features, projected_depth, validity_indicator, viewing_direction


# deprecated
@jaxtyped(typechecker=beartype)
def project_voxel_grid_to_images(
    voxel_grid: Float[Tensor, "3 X Y Z"],
    images: Float[Tensor, "I 3 H W"],
    transformations: Float[Tensor, "I 3 4"],
    T_cw: Float[Tensor, "I 4 4"],
    add_positional_encoding: bool = False,
    # channels: int = 12,
    add_projected_depth: bool = False,
    add_validity_indicator: bool = False,
    add_viewing_directing: bool = False,
    seq_len: Optional[int] = None,
) -> Float[Tensor, "F X Y Z"]:
    """
    I = Images
    F = I * (3 + add_projected_depth + add_validity_indicator + add_viewing_directing * 3)
    images: expected to be normalized [0, 1] float images C, H, W

    """
    _3, X, Y, Z = voxel_grid.shape

    # reshape to points
    points = rearrange(voxel_grid, "C X Y Z -> (X Y Z) C")

    # reshape points [num_points, 3] so that we have [num_points, num_images, 3]
    num_points = points.shape[0]
    num_images, _3, height, width = images.shape

    points = points.reshape(num_points, 1, 3).repeat(1, num_images, 1)

    if add_viewing_directing:
        R_cw = T_cw[:, :3, :3]
        t_cw = T_cw[:, :3, 3]
        R_wc = torch.transpose(R_cw, 1, 2)
        t_wc = -torch.matmul(R_wc, t_cw.unsqueeze(-1)).squeeze(-1)
        viewing_direction = points - t_wc
        viewing_direction = viewing_direction / torch.linalg.norm(
            viewing_direction, dim=2
        ).unsqueeze(-1)

    # convert to homographic coordinates
    points = torch.cat(
        [points, torch.full((num_points, num_images, 1), 1).to(points)], dim=-1
    )
    # and convert it to "matrix" (4x1) (num_points, num_images, 4, 1)
    points = rearrange(points, "n i d -> n i d 1")

    # perform the matmul with broadcasting (num_points, num_images, 4, 4) x (num_points, num_images, 4, 1) -> (num_points, num_images, 3, 1)
    transformed_points = torch.matmul(transformations, points)

    # remove last dimension (num_points, num_images, 3)
    transformed_points = rearrange(transformed_points, "n i d 1 -> n i d")

    if add_projected_depth:
        transformed_points_without_K = torch.matmul(T_cw[:, :3, :], points)
        transformed_points_without_K = rearrange(
            transformed_points_without_K, "n i d 1 -> n i d"
        )
        projected_depth = transformed_points_without_K[:, :, 2]

    # perform perspective division
    transformed_points = transformed_points[:, :, :3] / transformed_points[
        :, :, 2
    ].unsqueeze(-1)
    # and drop the last dimension
    transformed_points = transformed_points[:, :, :2]

    # now we need to mask by the bounding box of the image and replace with a constant fill (-inf)
    fill_value = -1
    mask = (
        (transformed_points[:, :, 0] >= 0)
        & (transformed_points[:, :, 0] < width)
        & (transformed_points[:, :, 1] >= 0)
        & (transformed_points[:, :, 1] < height)
    )
    full_mask = mask.reshape(-1, num_images, 1).repeat(1, 1, 2)
    valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

    # now sample the image at the valid pixels
    # the way this is done currently all of the "invalid" uvs will be 0, but we only use the valid ones later so its fine
    grid = rearrange(valid_pixels, "p n two -> n 1 p two", two=2)
    # normalize in last dimension to range [-1, 1]
    grid = (grid / torch.tensor([width, height]).to(grid)) * 2 - 1

    sampled = torch.nn.functional.grid_sample(images, grid)
    sampled = rearrange(sampled, "images channels 1 points -> points images channels")

    sampled[~mask] = fill_value

    if add_projected_depth:
        projected_depth = projected_depth.unsqueeze(-1)
        sampled = torch.cat([sampled, projected_depth], dim=-1)

    if add_validity_indicator:
        validity_indicator = mask.float().unsqueeze(-1)
        sampled = torch.cat([sampled, validity_indicator], dim=-1)

    if add_viewing_directing:
        sampled = torch.cat([sampled, viewing_direction], dim=-1)

    # get final number of channels
    # contains 3 for the color values and add_projected_depth + add_validity_indicator
    final_channels = sampled.shape[-1]

    # reshape to (num_points, num_images * 3)
    sampled = rearrange(sampled, "points images channels -> points (images channels)")

    # pad to final_channels * seq_len with -1
    if seq_len is not None:
        sampled = torch.cat(
            [
                sampled,
                torch.full(
                    (num_points, final_channels * seq_len - sampled.shape[1]),
                    fill_value,
                ).to(sampled.device),
            ],
            dim=-1,
        )

    sampled = rearrange(sampled, "(X Y Z) F -> F X Y Z", X=X, Y=Y, Z=Z)

    return sampled
