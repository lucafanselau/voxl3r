#! voxel grid and their transformations

import torch
from torch import Tensor
from jaxtyping import Float
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import roma


VoxelGridData = Float[Tensor, "R R R F"]


class CoordinateSystemType(Enum):
    WORLD = "world"
    CHUNK = "chunk"
    IMAGE = "image"


@dataclass
class CoordinateSystem:
    type: CoordinateSystemType
    reference_image: Optional[str] = None

    # transformation (always world to this)
    transform: Optional[roma.Rigid] = None

    def __post_init__(self):
        if self.type in [CoordinateSystemType.CHUNK, CoordinateSystemType.IMAGE]:
            if self.reference_image is None:
                raise ValueError(
                    f"reference_image must be provided for coordinate system type {self.type}"
                )


class VoxelGrid:
    coordinate_system: CoordinateSystem
    data: VoxelGridData
    pitch: float

    def __init__(
        self, data: VoxelGridData, coordinate_system: CoordinateSystem, pitch: float
    ):
        self.data = data
        self.coordinate_system = coordinate_system
        self.pitch = pitch

    # @staticmethod
    # def from_data(data: VoxelGridData):
    #     return VoxelGrid(data)

    def apply_transform(self, transform: roma.Rigid) -> "VoxelGrid":
        pass

    def to_points(self) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N F"]]:
        # get all the possible indices
        all_indices = torch.cartesian_prod(
            *[torch.arange(self.data.shape[i]) for i in range(3)]
        )
        # values = self.data[all_indices]

        # get the points as float, applying the pitch
        points = (all_indices.float() + 0.5) * self.pitch

        return points  # values

    @staticmethod
    def world_sphere(
        radius: float, base: Float[Tensor, "3 1"], pitch: float, resolution: int
    ) -> "VoxelGrid":
        # create an occupancy grid of a sphere in the world coordinate system

        # create a voxel grid
        data = torch.zeros(resolution, resolution, resolution, 1)
        # init an instance of the voxel grid
        voxel_grid = VoxelGrid(
            data, CoordinateSystem(CoordinateSystemType.WORLD), pitch
        )

        # create a sphere
        # get points
        points = voxel_grid.to_points()

        # apply the transform
        mask = (points - base).norm(dim=-1) <= radius

        # set the values
        voxel_grid.data[mask] = 1.0

        # return the voxel grid
        return voxel_grid

    def visualize(self):
        # visualize the voxel grid using trimesh and pyvista

        pass


def main():
    voxel_grid = VoxelGrid.world_sphere(1.0, torch.tensor([0.0, 0.0, 1.0]), 0.01, 200)
    pass


if __name__ == "__main__":
    main()
