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

    # Add new visualization options
    clip_plane_normal: Optional[Tuple[float, float, float]] = None
    clip_plane_origin: Optional[Tuple[float, float, float]] = None
    use_opacity_array: bool = False
    opacity_threshold: float = 0.5  # For thresholding when using opacity arrays


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
        """Create a PyVista structured grid from voxel values.

        The grid is created so that each cell corresponds to one voxel in the input.
        For a grid of N cells in any dimension, we need N+1 points to define the cell boundaries.
        """
        depth, height, width = values.shape[-3:]
        print(f"Input voxel dimensions: {depth}x{height}x{width}")

        # Create coordinate arrays with one more point than cells in each dimension
        # This is because points define corners of cells
        x = np.arange(width + 1) * self.config.grid_spacing + origin[0]
        y = np.arange(height + 1) * self.config.grid_spacing + origin[1]
        z = np.arange(depth + 1) * self.config.grid_spacing + origin[2]

        # Create mesh grid of points
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        print(f"Created grid with dimensions: {grid.dimensions}")
        print(f"Number of cells: {grid.number_of_cells}")
        print(f"Expected number of cells: {depth * height * width}")

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

    def visualize_with_opacity_array(
        self,
        voxels: Float[Tensor, "channels depth height width"],
        opacity_values: Optional[Float[Tensor, "depth height width"]] = None,
        show: bool = True,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> pv.StructuredGrid:
        """Visualize voxel grid using volume rendering."""
        voxels = self._tensor_to_numpy(voxels)

        # Create grid with specified origin
        grid = self._create_voxel_grid(voxels, origin=origin)

        # Handle opacity values
        if opacity_values is None:
            opacity = self._tensor_to_numpy(voxels[0])
        else:
            opacity = self._tensor_to_numpy(opacity_values)

        print(f"Opacity array shape: {opacity.shape}")
        print(f"Number of non-zero opacity values: {np.sum(opacity > 0)}")
        print(f"Total number of cells: {opacity.size}")
        print(
            f"Percentage of visible cells: {(np.sum(opacity > 0)/opacity.size)*100:.2f}%"
        )

        # Normalize opacity to 0-1 range if needed
        if opacity.max() > 1.0:
            opacity = opacity / opacity.max()

        # Threshold opacity values
        opacity[opacity < self.config.opacity_threshold] = 0.0

        # Add data to grid using cell data
        if voxels.shape[0] == 3:  # RGB data
            rgb = np.moveaxis(voxels, 0, -1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0

            # Convert RGB to scalar values for volume rendering
            # Using luminance as scalar value: Y = 0.2126 R + 0.7152 G + 0.0722 B
            scalars = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            grid.cell_data["values"] = scalars.flatten()

            # Create color transfer function based on RGB values
            color_tf = np.zeros((256, 3))
            for i in range(256):
                # Map scalar value back to RGB
                mask = (scalars >= i / 256) & (scalars < (i + 1) / 256)
                if mask.any():
                    color_tf[i] = np.mean(rgb[mask], axis=0)
                else:
                    color_tf[i] = color_tf[i - 1] if i > 0 else [0, 0, 0]

            # Create opacity transfer function
            opacity_tf = np.linspace(0, 1, 256)
            opacity_tf[0] = 0  # Make background transparent

            print("Adding RGB volume...")
            volume = self.plotter.add_volume(
                grid,
                scalars="values",
                opacity=opacity.flatten(),
                cmap=color_tf,
                opacity_unit_distance=self.config.grid_spacing,
                shade=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.5,
            )
        else:
            # For scalar data, use direct volume rendering
            grid.cell_data["values"] = voxels[0].flatten()

            print("Adding scalar volume...")
            volume = self.plotter.add_volume(
                grid,
                scalars="values",
                opacity=opacity.flatten(),
                cmap=self.config.cmap,
                opacity_unit_distance=self.config.grid_spacing,
                shade=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.5,
            )

        print(f"Final volume dimensions: {volume.dimensions}")
        print(f"Final volume cells: {volume.n_cells}")

        if show:
            self.show()

        return grid

    def visualize_with_point_masking(
        self,
        voxels: Float[Tensor, "channels depth height width"],
        mask: Optional[Float[Tensor, "depth height width"]] = None,
        show: bool = True,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> pv.StructuredGrid:
        """Visualize voxel grid using volume rendering with masking."""
        voxels = self._tensor_to_numpy(voxels)

        # Create grid with specified origin
        grid = self._create_voxel_grid(voxels, origin=origin)

        # Handle mask
        if mask is None:
            mask = self._tensor_to_numpy(voxels[0] > 0)
        else:
            mask = self._tensor_to_numpy(mask)

        print(f"Mask shape: {mask.shape}")
        print(f"Number of True values in mask: {mask.sum()}")
        print(f"Total number of cells to mask: {mask.size}")
        print(
            f"Percentage of cells that should be shown: {(mask.sum()/mask.size)*100:.2f}%"
        )

        # Convert mask to opacity values
        opacity = mask.astype(float)

        if voxels.shape[0] == 3:  # RGB data
            rgb = np.moveaxis(voxels, 0, -1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0

            # Convert RGB to scalar values for volume rendering
            scalars = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            grid.cell_data["values"] = scalars.flatten()

            # Create color transfer function based on RGB values
            color_tf = np.zeros((256, 3))
            for i in range(256):
                mask = (scalars >= i / 256) & (scalars < (i + 1) / 256)
                if mask.any():
                    color_tf[i] = np.mean(rgb[mask], axis=0)
                else:
                    color_tf[i] = color_tf[i - 1] if i > 0 else [0, 0, 0]

            print("Adding RGB volume...")
            volume = self.plotter.add_volume(
                grid,
                scalars="values",
                opacity=opacity.flatten(),
                cmap=color_tf,
                opacity_unit_distance=self.config.grid_spacing,
                shade=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.5,
            )
        else:
            grid.cell_data["values"] = voxels[0].flatten()

            print("Adding scalar volume...")
            volume = self.plotter.add_volume(
                grid,
                scalars="values",
                opacity=opacity.flatten(),
                cmap=self.config.cmap,
                opacity_unit_distance=self.config.grid_spacing,
                shade=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.5,
            )

        if show:
            self.show()

        return grid

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
    opacity_values: Optional[Float[Tensor, "batch depth height width"]] = None,
    mask: Optional[Float[Tensor, "batch depth height width"]] = None,
) -> None:
    """Convenience function to quickly visualize voxel grids."""
    if config is None:
        config = VoxelVisualizerConfig()

    visualizer = VoxelGridVisualizer(config)
    batch_size = len(voxels)

    # Determine grid arrangement for batched visualization
    if config.grid_arrangement is None:
        rows = int(np.ceil(np.sqrt(batch_size)))
        cols = int(np.ceil(batch_size / rows))
    else:
        rows, cols = config.grid_arrangement

    if len(voxels.shape) == 4:  # Single grid
        if mask is not None:
            visualizer.visualize_with_point_masking(voxels, mask=mask)
        elif opacity_values is not None:
            visualizer.visualize_with_opacity_array(
                voxels, opacity_values=opacity_values
            )
        else:
            visualizer.visualize_single_grid(voxels)
    else:  # Batch of grids
        if mask is not None:
            for i in range(batch_size):
                # Calculate grid position
                row = i // cols
                col = i % cols

                # Calculate offset for this grid
                offset = (
                    col
                    * config.grid_spacing
                    * voxels.shape[-1]
                    * config.spacing_factor,
                    row
                    * config.grid_spacing
                    * voxels.shape[-2]
                    * config.spacing_factor,
                    0,
                )

                visualizer.visualize_with_point_masking(
                    voxels[i], mask=mask[i], show=False, origin=offset
                )
            visualizer.show()
        elif opacity_values is not None:
            for i in range(batch_size):
                # Calculate grid position
                row = i // cols
                col = i % cols

                # Calculate offset for this grid
                offset = (
                    col
                    * config.grid_spacing
                    * voxels.shape[-1]
                    * config.spacing_factor,
                    row
                    * config.grid_spacing
                    * voxels.shape[-2]
                    * config.spacing_factor,
                    0,
                )

                visualizer.visualize_with_opacity_array(
                    voxels[i],
                    opacity_values=opacity_values[i],
                    show=False,
                    origin=offset,
                )
            visualizer.show()
        else:
            visualizer.visualize_batch(voxels)

    if save_path is not None:
        visualizer.save_screenshot(save_path)

    visualizer.close()
