from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvista as pv

from utils.transformations import invert_pose, project_points


def visualize_images(image_tensors, titles=None, cols=3, figsize=(15, 5)):
    """
    Visualizes a list of image tensors, handling both color and grayscale images.

    Args:
        image_tensors (list or torch.Tensor): List of image tensors or a single image tensor.
                                             Each image tensor can be:
                                                - 3D (C x H x W) for color or single-channel
                                                - 3D (H x W x C) for color or single-channel
                                                - 2D (H x W) for single-channel
        titles (list of str, optional): Titles for each image. Defaults to ['Image 1', 'Image 2', ...].
        cols (int, optional): Number of columns in the plot grid. Defaults to 3.
        figsize (tuple, optional): Figure size. Defaults to (15, 5).
    """
    # Ensure image_tensors is a list
    if isinstance(image_tensors, torch.Tensor):
        image_tensors = [image_tensors]
    elif not isinstance(image_tensors, (list, tuple)):
        image_tensors = [image_tensors]
        
    num_images = len(image_tensors)
    
    # Generate default titles if not provided
    if titles is None:
        titles = [f'Image {i+1}' for i in range(num_images)]
    elif len(titles) != num_images:
        raise ValueError("Number of titles must match number of images")
    
    # Calculate number of rows needed
    rows = (num_images + cols - 1) // cols
    
    # Create a new figure
    plt.figure(figsize=figsize)
    
    for idx, image_tensor in enumerate(image_tensors):
        plt.subplot(rows, cols, idx + 1)
        
        # Determine the format and convert to numpy
        if image_tensor.dim() == 3:
            if image_tensor.shape[0] == 3:
                # C x H x W (e.g., RGB)
                image_np = image_tensor.permute(1, 2, 0).numpy()
            elif image_tensor.shape[2] == 3:
                # H x W x C (e.g., RGB)
                image_np = image_tensor.numpy()
            elif image_tensor.shape[0] == 1:
                # 1 x H x W (grayscale with explicit channel)
                image_np = image_tensor.squeeze(0).numpy()
            elif image_tensor.shape[2] == 1:
                # H x W x 1 (grayscale with explicit channel)
                image_np = image_tensor.squeeze(2).numpy()
            else:
                raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")
        elif image_tensor.dim() == 2:
            image_np = image_tensor.numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")
        

        
        # Determine if the image is grayscale or color for plotting
        if image_np.ndim == 2:
            # Grayscale image
            plt.imshow(image_np, cmap='gray')
        else:
            # Color image
            # Uncomment the following line if images are in BGR format (common with OpenCV)
            # image_np = image_np[:, :, ::-1]
            plt.imshow(image_np)
        
        # Set the title and remove axes
        plt.title(titles[idx])
        plt.axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
def get_camera_corners(cam_params, plane_distance=1.0):
    """
    Get the 3D coordinates of the four image corners in camera coordinates at a given plane distance.

    Args:
        cam_params (dict): Camera parameters containing 'K' (3x3 intrinsic matrix).
        plane_distance (float): Distance of the image plane from the camera center along the Z-axis.

    Returns:
        np.ndarray: Array of shape (4, 3) with the 3D coordinates of the four corners
                    in camera coordinates at the specified plane distance.
    """
    # Define the four corners of the image in pixel coordinates
    pixel_corners = np.array([
        [0, 0],                        # Top-left corner
        [cam_params['width'], 0],              # Top-right corner
        [cam_params['width'], cam_params['height']],   # Bottom-right corner
        [0, cam_params['height']],             # Bottom-left corner
    ])  # Shape: [4, 2]

    # Convert pixel coordinates to homogeneous coordinates
    homogeneous_pixel_corners = np.hstack([pixel_corners, np.ones((4, 1))])  # Shape: [4, 3]

    # Compute the inverse of the intrinsic matrix K
    K_inv = np.linalg.inv(cam_params['K'])

    # Convert to normalized camera coordinates
    corners_cam = (K_inv @ homogeneous_pixel_corners.T).T  # Shape: [4, 3]

    # Scale the normalized coordinates to have Z = plane_distance
    corners_cam *= (plane_distance / corners_cam[:, 2:3])

    return corners_cam
    
def create_image_plane(cam_params, plane_distance):
    """
    Create a PyVista plane representing the image in 3D space with correct orientation.

    Args:
        cam_params (dict): Camera parameters containing 'R_cw', 't_cw', 'K'.
        plane_distance (float): Distance from the camera center to the plane.

    Returns:
        pv.PolyData: The plane positioned in 3D space.
    """
    corners_cam = get_camera_corners(cam_params, plane_distance)

    # Scale normalized coordinates to have Z = plane_distance
    corners_cam *= (plane_distance / corners_cam[:, 2:3])

    # Compute the center of the plane in camera coordinates
    center_cam = corners_cam.mean(axis=0)  # Shape: [3,]

    # The plane's normal vector in camera coordinates
    direction_cam = np.array([0, 0, 1])  # Assuming the plane is facing along the positive Z-axis

    # The size of the plane along i and j axes (in camera coordinates)
    i_size = np.linalg.norm(corners_cam[1] - corners_cam[0])  # Width of the plane
    j_size = np.linalg.norm(corners_cam[3] - corners_cam[0])  # Height of the plane

    # Create the plane in camera coordinates
    plane = pv.Plane(
        center=center_cam,
        direction=direction_cam,
        i_size=i_size,
        j_size=j_size,
        i_resolution=1,
        j_resolution=1,
    )

    # Compute the rotation matrix to align the plane's local axes with the camera's axes
    # Rotate the plane around the X-axis by 180 degrees to flip the Y-axis
    plane.rotate_x(180, point=center_cam, inplace=True)

    # Transform the plane from camera to world coordinates
    R_cw = cam_params['R_cw']
    t_cw = cam_params['t_cw'].flatten()
    R_wc, t_wc = invert_pose(R_cw, t_cw)

    # Create the transformation matrix
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_wc
    T_cam_world[:3, 3] = t_wc

    # Apply the transformation
    plane.transform(T_cam_world)

    # Assign texture coordinates (UV mapping)
    # The plane's texture coordinates need to be adjusted because of the flip
    plane.texture_map_to_plane(inplace=False)

    return plane

def visualize_mesh_without_vertices(mesh_path, label_dict, remove_labels):
    # Load the mesh
    mesh = pv.read(mesh_path)

    # Collect vertex indices to remove
    vertices_to_remove = set()
    for label in remove_labels:
        if label in label_dict:
            vertices_to_remove.update(label_dict[label])
    
    # Convert the set to a sorted list or array for indexing
    vertices_to_remove = np.array(sorted(vertices_to_remove))

    # Create a mask to keep only vertices that are not in `vertices_to_remove`
    mask = np.ones(mesh.n_points, dtype=bool)
    mask[vertices_to_remove] = False

    # Extract the remaining vertices, which excludes the vertices in `vertices_to_remove`
    new_mesh = mesh.extract_points(mask, adjacent_cells=False)

    # Create a plotter object
    plotter = pv.Plotter()

    # Add the mesh with vertices removed to the plotter
    plotter.add_mesh(new_mesh, color='white', opacity=1.0)

    # Show the coordinate axes
    plotter.show_axes()

    # Show the plot
    plotter.show()


def visualize_mesh(mesh, images=None, camera_params_list=None, point_coords=None, heat_values=None, plane_distance=0.1, offsets=[]):
    """
    Visualize a mesh along with images projected into 3D space according to their camera parameters,
    and optionally plot a point at a specified location.

    Args:
        mesh (str or pyvista.DataSet ): Path to the .ply mesh file.
        images (list of str, optional): List of image file paths.
        camera_params_list (list of dict, optional): List of camera parameters for each image.
            Each dict should contain:
                - 'R_cw': Rotation matrix from world to camera coordinates (3x3 numpy array)
                - 't_cw': Translation vector from world to camera coordinates (3x1 numpy array)
                - 'K': Camera intrinsic matrix (3x3 numpy array)
        point_coords (array-like or list of array-like, optional): Coordinates of the point(s) to plot.
            Can be a single point [x, y, z] or a list of points.
        plane_distance (float, optional): Distance of the image planes from their camera center.
        offsets (list of float, optional): List of offsets for each image plane along the Z-axis.
    """
    # Load the mesh
    if isinstance(mesh, Path):
        mesh = pv.read(mesh)

    # Create a plotter object
    plotter = pv.Plotter()

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, color='white', opacity=1.0)

    # For each image, create a textured plane and add it to the plotter
    if images is not None and camera_params_list is not None:
        for img_path, cam_params in zip(images, camera_params_list):
            
            R_cw = cam_params['R_cw']
            t_cw = cam_params['t_cw'].flatten()
            R_wc, t_wc = invert_pose(R_cw, t_cw)
            
            # Draw the camera center
            c_point = pv.PolyData(t_wc)
            plotter.add_mesh(c_point, color='grey', point_size=10, render_points_as_spheres=True)
            
            # Draw image plane
            corners_cam = get_camera_corners(cam_params, plane_distance + (offsets[-1] if len(offsets) > 0 else 0))
            for corner in corners_cam:
                corner = R_wc @ corner + t_wc
                line_points = np.array([t_wc, corner])
                line = pv.lines_from_points(line_points)
                plotter.add_mesh(line, color='black', line_width=4)
                
            
            # Draw the image 
            texture = pv.read_texture(img_path)
            plane = create_image_plane(cam_params, plane_distance=plane_distance)
            plotter.add_mesh(plane, texture=texture)

            # Draw the offset planes
            while len(offsets) != 0:
                plane_distance = plane_distance + offsets.pop(0)
                # Create the plane in 3D space
                plane = create_image_plane(cam_params, plane_distance=plane_distance)
                # Add the textured plane to the plotter
                plotter.add_mesh(plane, texture=texture)
                
            # Add a line from the camera center to the 3D point
            if point_coords is not None:
                point_coords = np.asarray(point_coords).reshape(-1, 3)
                if point_coords.shape[0] == 1:
                    P_world = point_coords[0]
                    line_points = np.array([t_wc, P_world])
                    line = pv.lines_from_points(line_points)
                    plotter.add_mesh(line, color='black', line_width=4)

    if point_coords is not None:
        point_coords = np.asarray(point_coords).reshape(-1, 3)
        points = pv.PolyData(point_coords)
        
        # Determine point size
        p_size = 15 if point_coords.shape[0] == 1 else 10
        
        if heat_values is not None:
            heat_values = np.asarray(heat_values).flatten()
            if heat_values.shape[0] != point_coords.shape[0]:
                raise ValueError("Length of heat_values must match number of point_coords.")
            
            points['Heat'] = heat_values
            
            plotter.add_mesh(
                points, 
                scalars='Heat', 
                cmap='hot',
                point_size=p_size, 
                render_points_as_spheres=True,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Heat', 'shadow': True}
            )
        else:
            plotter.add_mesh(
                points, 
                color='red', 
                point_size=p_size, 
                render_points_as_spheres=True
            )

    # Show the coordinate axes
    plotter.show_axes()

    # Show the plot
    plotter.show()

