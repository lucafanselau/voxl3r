from typing import Optional, Tuple
from einops import rearrange, repeat
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

def to_homogeneous(x: Float[Tensor, "... 3"]) -> Float[Tensor, "... 4"]:
    return torch.cat([x, torch.ones_like(x[..., 0:1])], dim=-1)

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
def batch_project_voxel_grid_to_images_seperate(
    voxel_grid: Float[Tensor, "B 3 X Y Z"],
    images: Float[Tensor, "B I F H W"],
    transformations: Float[Tensor, "B I 3 4"],
    T_cw: Float[Tensor, "B I 4 4"],
    grid_sampling_mode: Optional[str] = "bilinear",
) -> Tuple[Float[Tensor, "B I F X Y Z"], Float[Tensor, "B I 1 X Y Z"], Float[Tensor, "B I 1 X Y Z"], Float[Tensor, "B I 3 X Y Z"]]:
    """
    I = Images
    F = I * (3 + add_projected_depth + add_validity_indicator + add_viewing_directing * 3)
    images: expected to be normalized [0, 1] float images C, H, W

    returns rgb_features, projected_depth, validity_indicator, viewing_direction

    """
    B, _3, X, Y, Z = voxel_grid.shape

    # reshape to points
    points = rearrange(voxel_grid, "B C X Y Z -> B (X Y Z) C")

    num_points = points.shape[-2]
    B, num_images, _3, height, width = images.shape

    # points is [B, grid_size**3, 3] we want [B, grid_size**3, num_images, 3]
    points = repeat(points, "B N C -> B N I C", I=num_images)

    R_cw = T_cw[..., :3, :3]
    t_cw = T_cw[..., :3, 3]
    R_wc = torch.transpose(R_cw, -1, 2)
    t_wc = -torch.matmul(R_wc, t_cw.unsqueeze(-1)).squeeze(-1)
    # unsqueeze to [B, N (eg. 1), num_images, 3]
    viewing_direction = points - t_wc.unsqueeze(1)
    viewing_direction = viewing_direction / torch.linalg.norm(
        viewing_direction, dim=-1
    ).unsqueeze(-1)

    # viewing_direction is [B, N, num_images, 3]

    # convert to homographic coordinates
    points = to_homogeneous(points)
    # and convert it to "matrix" (4x1) (num_points, num_images, 4, 1)
    points = points.unsqueeze(-1)

    # make transformations [B, N, num_images, 3, 4]
    transformations = rearrange(transformations, "B I THREE FOUR -> B 1 I THREE FOUR")

    # perform the matmul with broadcasting (batch, 1, num_images, 3, 4) x (batch, num_points, num_images, 4, 1) -> (batch, num_points, num_images, 3, 1)
    transformed_points = torch.matmul(transformations, points)

    # remove last dimension (num_points, num_images, 3)
    transformed_points = rearrange(transformed_points, "... 1 -> ...")

    T_cw = rearrange(T_cw, "B I FOUR F -> B 1 I FOUR F")

    transformed_points_without_K = torch.matmul(T_cw[..., :3, :], points)
    transformed_points_without_K = rearrange(
        transformed_points_without_K, "... 1 -> ..."
    )
    projected_depth = transformed_points_without_K[..., 2].unsqueeze(-1)

    # perform perspective division
    transformed_points = transformed_points[..., :3] / transformed_points[..., 2].unsqueeze(-1)
    # and drop the last dimension
    transformed_points = transformed_points[..., :2]

    # now we need to mask by the bounding box of the image and replace with a constant fill (-inf)
    mask = (
        (transformed_points[..., 0] >= 0)
        & (transformed_points[..., 0] < width)
        & (transformed_points[..., 1] >= 0)
        & (transformed_points[..., 1] < height)
    )
    # full_mask = mask.reshape(-1, num_images, 1).repeat(1, 1, 2)
    # valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

    # since we are targeting the 2d use case of grid_sample, we need to merge batches and instances
    images_flat = rearrange(images, "B I F H W -> (B I) F H W")
    # the grid has to be (N,H_out,W_out,2)
    # here we work with a "sample grid" that is just a line, eg H_out = 1, W_out = grid_size**3 (32**3)
    transformed_points_flat = rearrange(transformed_points, "B P I TWO -> (B I) 1 P TWO", P=num_points)

    # normalize to [-1, 1]
    # grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]
    grid = (transformed_points_flat / torch.tensor([width, height]).to(transformed_points_flat)) * 2 - 1

    # amount of valid pixels: ((grid > -1) & (grid < 1)).sum() / grid.numel()

    sampled = torch.nn.functional.grid_sample(images_flat, grid, align_corners=True, mode=grid_sampling_mode, padding_mode="zeros")

    smeared_features = rearrange(sampled, "(B I) F 1 (X Y Z) -> B I F X Y Z", B=B, X=X, Y=Y, Z=Z)
    projected_depth = rearrange(projected_depth, "B (X Y Z) I 1 -> B I 1 X Y Z", X=X, Y=Y, Z=Z)
    validity_indicator = rearrange(mask, "B (X Y Z) I -> B I 1 X Y Z", X=X, Y=Y, Z=Z)
    viewing_direction = rearrange(viewing_direction, "B (X Y Z) I F -> B I F X Y Z", X=X, Y=Y, Z=Z)


    # smeared: Tuple[Float[Tensor, "B I F X Y Z"], 
    # depth: Float[Tensor, "B I 1 X Y Z"], 
    # validity_indicator: Float[Tensor, "B I 1 X Y Z"], 
    # viewing_direction: Float[Tensor, "B I 3 X Y Z"]]:
    return smeared_features, projected_depth, validity_indicator.float(), viewing_direction