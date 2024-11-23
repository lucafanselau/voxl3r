from typing import Optional
from einops import rearrange
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import lightning as pl
from torch import nn
from dataset import SceneDataset, SceneDatasetTransformToTorch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from positional_encodings.torch_encodings import PositionalEncoding3D

from utils.chunking import create_chunk, mesh_2_voxels


# Implementation of positional encoding for

# def get_emb(sin_inp):
#     """
#     Gets a base embedding for one dimension with sin and cos intertwined
#     """
#     emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
#     return torch.flatten(emb, -2, -1)


# class PositionalEncoding3D(nn.Module):
#     def __init__(self, channels, dtype_override=None):
#         """
#         :param channels: The last dimension of the tensor you want to apply pos emb to.
#         :param dtype_override: If set, overrides the dtype of the output embedding.
#         """
#         super(PositionalEncoding3D, self).__init__()
#         self.org_channels = channels
#         channels = int(np.ceil(channels / 6) * 2)
#         if channels % 2:
#             channels += 1
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
#         self.register_buffer("inv_freq", inv_freq)
#         self.register_buffer("cached_penc", None, persistent=False)
#         self.dtype_override = dtype_override
#         self.channels = channels

#     def forward(self, tensor):
#         """
#         :param tensor: A 2d tensor of size (batch_size, 3)
#         :return: Positional Encoding Matrix of size (batch_size, self.channels * 3)
#         """

#         # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
#         #     return self.cached_penc

#         # self.cached_penc = None
#         # batch_size, x, y, z, orig_ch = tensor.shape
#         # pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
#         # pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
#         # pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
#         pos_x = tensor[:, 0]
#         pos_y = tensor[:, 1]
#         pos_z = tensor[:, 2]
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
#         sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
#         sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
#         emb_x = get_emb(sin_inp_x)
#         emb_y = get_emb(sin_inp_y)
#         emb_z = get_emb(sin_inp_z)
#         # emb = torch.zeros(
#         #     (x, y, z, self.channels * 3),
#         #     device=tensor.device,
#         #     dtype=(
#         #         self.dtype_override if self.dtype_override is not None else tensor.dtype
#         #     ),
#         # )
#         # emb[:, :, :, : self.channels] = emb_x
#         # emb[:, :, :, self.channels : 2 * self.channels] = emb_y
#         # emb[:, :, :, 2 * self.channels :] = emb_z
#         emb = torch.cat([emb_x, emb_y, emb_z], dim=1)

#         # self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
#         return emb


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

#@jaxtyped(typechecker=beartype)
def project_voxel_grid_to_images_seperate(
    voxel_grid: Float[Tensor, "3 X Y Z"],
    images: Float[Tensor, "I 3 H W"],
    transformations: Float[Tensor, "I 3 4"],
    T_cw: Float[Tensor, "I 4 4"],
) -> Float[Tensor, "F X Y Z"] :
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
    rgb_features = rearrange(sampled, "images channels 1 points -> points images channels")

    rgb_features[~mask] = fill_value
    validity_indicator = mask.float().unsqueeze(-1)
    
    rgb_features = rearrange(rgb_features, "points images channels -> points (images channels)")
    rgb_features = rearrange(rgb_features, "(X Y Z) F -> F X Y Z", X=X, Y=Y, Z=Z)
    
    projected_depth = rearrange(projected_depth, "points images 1 -> points (images 1)")
    projected_depth = rearrange(projected_depth, "(X Y Z) F-> F X Y Z", X=X, Y=Y, Z=Z)
    
    validity_indicator = rearrange(validity_indicator, "points images 1 -> points (images 1)")
    validity_indicator = rearrange(validity_indicator, "(X Y Z) F -> F X Y Z", X=X, Y=Y, Z=Z)
    
    viewing_direction = rearrange(viewing_direction, "points images channels -> points (images channels)")
    viewing_direction = rearrange(viewing_direction, "(X Y Z) F -> F X Y Z", X=X, Y=Y, Z=Z)
    
    return rgb_features, projected_depth, validity_indicator, viewing_direction

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
