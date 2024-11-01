import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_pose(R_cw, t_cw):
    """
    Inverts the camera pose from camera-to-world to world-to-camera coordinates.

    Args:
        R_cw (ndarray): Rotation matrix from camera to world coordinates (shape: [3, 3]).
        t_cw (ndarray): Translation vector from camera to world coordinates (shape: [3, 1]).

    Returns:
        R_wc (ndarray): Rotation matrix from world to camera coordinates (shape: [3, 3]).
        t_wc (ndarray): Translation vector from world to camera coordinates (shape: [3, 1]).
    """
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    return R_wc, t_wc

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a rotation matrix.

    Args:
        qw, qx, qy, qz (float): Quaternion components.

    Returns:
        R_matrix (ndarray): Rotation matrix (shape: [3, 3]).
    """
    r = R.from_quat([qx, qy, qz, qw])
    R_matrix = r.as_matrix()
    return R_matrix

def project_points(P_world, K, R_cw, t_cw):
    """
    Projects multiple 3D world points into the image plane.

    Args:
        P_world (ndarray): 3D points in world coordinates (shape: [3, N]).
        K (ndarray): Intrinsic camera matrix (shape: [3, 3]).
        R_cw (ndarray): Rotation matrix from world to camera coordinates (shape: [3, 3]).
        t_cw (ndarray): Translation vector from camera center to world origin (shape: [3, 1]).

    Returns:
        uvs (ndarray): Projected pixel coordinates (shape: [2, M]).
        valid_indices (ndarray): Indices of points that are in front of the camera.
    """
    # ensures broadcasting works correctly
    t_cw = t_cw.reshape(3, 1)
    # Transform the points to camera coordinates
    P_camera = R_cw @ P_world + t_cw  # Shape: [3, N]

    # Check if the points are in front of the camera (Z > 0)
    in_front = P_camera[2, :] > 0  # Boolean array of shape [N]

    # Keep only points that are in front of the camera
    P_camera = P_camera[:, in_front]  # Shape: [3, M], where M <= N

    if P_camera.shape[1] == 0:
        return None, None  # No points are in front of the camera

    # Normalize by Z (depth)
    homogeneous_coord = P_camera / P_camera[2, :]

    # Project onto image plane
    pixel_coords = K @ homogeneous_coord  # Shape: [3, M]

    # Extract u and v coordinates
    u = pixel_coords[0, :]
    v = pixel_coords[1, :]

    # Combine u and v into a single array
    uvs = np.vstack((u, v))  # Shape: [2, M]

    # Get the indices of valid points
    valid_indices = np.where(in_front)[0]

    return uvs, valid_indices
