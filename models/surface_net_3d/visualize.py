from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class VoxelVisualizerConfig:
    """Configuration for voxel grid visualization."""

    # Visualization settings
    opacity: float = 0.5
    show_edges: bool = True
    edge_color: str = "black"
    cmap: str = "viridis"
    background_color: str = "white"
    window_size: Tuple[int, int] = (1024, 768)

    # Grid settings
    grid_spacing: float = 1.0
    grid_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Camera settings
    camera_position: Optional[Tuple[float, float, float]] = None
    view_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Multi-grid layout
    grid_arrangement: Optional[Tuple[int, int]] = None  # Rows, Cols for multiple grids
    spacing_factor: float = 1.2  # Space between grids


class VoxelGridVisualizer:
    def __init__(self, config: VoxelVisualizerConfig = VoxelVisualizerConfig()):
        """Initialize the visualizer with given configuration."""
        self.config = config
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
        )
        self.plotter.background_color = self.config.background_color

    def _tensor_to_numpy(
        self, tensor: Float[Tensor, "batch channels depth height width"]
    ) -> np.ndarray:
        """Convert PyTorch tensor to numpy array and ensure correct shape."""
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor

    def _create_voxel_grid(
        self, values: np.ndarray, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> pv.StructuredGrid:
        """Create a PyVista structured grid from voxel values."""
        depth, height, width = values.shape[-3:]

        # Create coordinate arrays
        x = np.arange(width) * self.config.grid_spacing + origin[0]
        y = np.arange(height) * self.config.grid_spacing + origin[1]
        z = np.arange(depth) * self.config.grid_spacing + origin[2]

        # Create mesh grid
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        return grid

    def visualize_single_grid(
        self, voxels: Float[Tensor, "channels depth height width"], show: bool = True
    ) -> pv.StructuredGrid:
        """Visualize a single voxel grid."""
        voxels = self._tensor_to_numpy(voxels)

        # Handle RGB values (3 channels)
        if voxels.shape[0] == 3:
            # Normalize RGB values to [0, 1]
            rgb = np.moveaxis(voxels, 0, -1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
        else:
            # Use first channel for non-RGB data
            rgb = voxels[0]

        # Create grid
        grid = self._create_voxel_grid(rgb)

        if voxels.shape[0] == 3:
            grid.point_data["RGB"] = rgb.reshape(-1, 3)
            self.plotter.add_mesh(
                grid,
                rgb=True,
                opacity=self.config.opacity,
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
            )
        else:
            grid.point_data["values"] = rgb.flatten()
            self.plotter.add_mesh(
                grid,
                scalars="values",
                cmap=self.config.cmap,
                opacity=self.config.opacity,
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
            )

        if show:
            self.show()

        return grid

    def visualize_batch(
        self,
        voxels: Float[Tensor, "batch channels depth height width"],
        show: bool = True,
    ) -> List[pv.StructuredGrid]:
        """Visualize a batch of voxel grids."""
        voxels = self._tensor_to_numpy(voxels)
        batch_size = voxels.shape[0]

        # Determine grid arrangement
        if self.config.grid_arrangement is None:
            rows = int(np.ceil(np.sqrt(batch_size)))
            cols = int(np.ceil(batch_size / rows))
        else:
            rows, cols = self.config.grid_arrangement

        grids = []
        for i in range(batch_size):
            # Calculate grid position
            row = i // cols
            col = i % cols

            # Calculate offset for this grid
            offset = (
                col
                * self.config.grid_spacing
                * voxels.shape[-1]
                * self.config.spacing_factor,
                row
                * self.config.grid_spacing
                * voxels.shape[-2]
                * self.config.spacing_factor,
                0,
            )

            # Create and add grid
            grid = self._create_voxel_grid(voxels[i], origin=offset)

            if voxels.shape[1] == 3:  # RGB
                rgb = np.moveaxis(voxels[i], 0, -1)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                grid.point_data["RGB"] = rgb.reshape(-1, 3)
                self.plotter.add_mesh(
                    grid,
                    rgb=True,
                    opacity=self.config.opacity,
                    show_edges=self.config.show_edges,
                    edge_color=self.config.edge_color,
                )
            else:
                grid.point_data["values"] = voxels[i, 0].flatten()
                self.plotter.add_mesh(
                    grid,
                    scalars="values",
                    cmap=self.config.cmap,
                    opacity=self.config.opacity,
                    show_edges=self.config.show_edges,
                    edge_color=self.config.edge_color,
                )

            grids.append(grid)

        if show:
            self.show()

        return grids

    def show(self):
        """Display the visualization."""
        if self.config.camera_position is not None:
            self.plotter.camera_position = self.config.camera_position
            self.plotter.camera.up = self.config.view_up

        self.plotter.show()

    def save_screenshot(self, filename: str):
        """Save the current view to an image file."""
        self.plotter.screenshot(filename)

    def close(self):
        """Close the plotter."""
        self.plotter.close()


def visualize_voxel_grids(
    voxels: Float[Tensor, "batch channels depth height width"],
    config: Optional[VoxelVisualizerConfig] = None,
    save_path: Optional[str] = None,
) -> None:
    """Convenience function to quickly visualize voxel grids."""
    if config is None:
        config = VoxelVisualizerConfig()

    visualizer = VoxelGridVisualizer(config)

    if len(voxels.shape) == 4:  # Single grid
        visualizer.visualize_single_grid(voxels)
    else:  # Batch of grids
        visualizer.visualize_batch(voxels)

    if save_path is not None:
        visualizer.save_screenshot(save_path)

    visualizer.close()
