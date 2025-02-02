from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
import torch


def interpolate_grid(
        original_voxel_grids,
        sample_coordiantes,
        pitch
        ):
    scaling_factor = (pitch / original_voxel_grids[0].pitch[0]).item()
    upscaled_coordinates = torch.nn.functional.interpolate(sample_coordiantes, scale_factor=scaling_factor, mode='trilinear' )
    B, C, X, Y, Z = upscaled_coordinates.shape
        
    original_transforms = torch.stack(
        [torch.from_numpy(voxel_grid.transform.copy()) for voxel_grid in original_voxel_grids]
    )
    original_extents = torch.stack(
        [torch.from_numpy(voxel_grid.extents.copy()) for voxel_grid in original_voxel_grids]
    )
    
    base_voxel_localtions = original_transforms[:, :3, 3]
    
    coordinate_points = rearrange(upscaled_coordinates, "B C X Y Z -> B (X Y Z) C")
    norm_coords = (coordinate_points - base_voxel_localtions.unsqueeze(1)) / original_extents.unsqueeze(1)
    norm_coords_minus_one = (2.0 * norm_coords - 1.0).float()
    
    
    original_occ_grids = torch.stack(
        [torch.from_numpy(voxel_grid.matrix.copy()) for voxel_grid in original_voxel_grids]
    )
    occ_grids = torch.nn.functional.grid_sample(
            rearrange(original_occ_grids.float(), "B X Y Z -> B 1 Z Y X"),
            rearrange(norm_coords_minus_one, "B (X Y Z) C -> B Z Y X C", C=C, X=X, Y=Y, Z=Z),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
    
    occ_grids = rearrange(occ_grids, "B 1 Z Y X -> B 1 X Y Z")
        # downscaling
    while(scaling_factor > 1):
        occ_grids = torch.nn.functional.interpolate(occ_grids, scale_factor=0.5, mode='trilinear')
        scaling_factor = scaling_factor / 2
        if scaling_factor % 1 != 0:
            raise ValueError("Scaling factor must be a power of 2")
        
    occ_grids[occ_grids > 0.0] = 1.0
    return occ_grids.bool()


@jaxtyped(typechecker=beartype)
def interpolate_grid_batch(
        sampled_voxel_grids: Float[Tensor, "B X1 Y1 Z1"],
        sampled_voxel_grid_extent: Float[Tensor, "B 3"],
        sampled_origin_base_voxel: Float[Tensor, "B 3"],
        sample_coordiante_grid: Float[Tensor, "B C X2 Y2 Z2"],
        sampled_voxel_grid_resolution: float,
        sample_coordiante_grid_resolution: float,
        ):
    scaling_factor = (sample_coordiante_grid_resolution / sampled_voxel_grid_resolution)
    upscaled_coordinate_grid = torch.nn.functional.interpolate(sample_coordiante_grid, scale_factor=scaling_factor, mode='trilinear', align_corners=False)
    B, C, X, Y, Z = upscaled_coordinate_grid.shape
    
    coordinate_points = rearrange(upscaled_coordinate_grid, "B C X Y Z -> B (X Y Z) C")
    norm_coords = (coordinate_points - sampled_origin_base_voxel.unsqueeze(1)) / sampled_voxel_grid_extent.unsqueeze(1)
    norm_coords_minus_one = (2.0 * norm_coords - 1.0).float()

    occ_grids = torch.nn.functional.grid_sample(
            rearrange(sampled_voxel_grids.float(), "B X Y Z -> B 1 Z Y X"),
            rearrange(norm_coords_minus_one, "B (X Y Z) C -> B Z Y X C", C=C, X=X, Y=Y, Z=Z),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )
    
    occ_grids = rearrange(occ_grids, "B 1 Z Y X -> B 1 X Y Z")
        # downscaling
    while(scaling_factor > 1):
        occ_grids = torch.nn.functional.interpolate(occ_grids, scale_factor=0.5, mode='trilinear', align_corners=False)
        scaling_factor = scaling_factor / 2
        if scaling_factor % 1 != 0:
            raise ValueError("Scaling factor must be a power of 2")
        
    occ_grids[occ_grids > 0.0] = 1.0
    return occ_grids.bool()


