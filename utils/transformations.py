import cv2
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
    t_cw = t_cw.reshape(3, 1)
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    T_wc = np.vstack((np.hstack((R_wc, t_wc)), np.array([0, 0, 0, 1])))
    return R_wc, t_wc, T_wc

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
    P_world = P_world.reshape(3, -1)
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

def undistort_image(image, K, dist_coeffs, alpha=0):
    """
    Undistort a single image using OpenCV's undistort function with optimized camera matrix.

    Args:
        image (numpy.ndarray): Image array in HxWxC format (uint8 or uint16).
        K (numpy.ndarray): Intrinsic camera matrix (3x3).
        dist_coeffs (numpy.ndarray): Distortion coefficients (e.g., [k1, k2, p1, p2, k3]).
        alpha (float): Free scaling parameter between 0 and 1.

    Returns:
        numpy.ndarray: Undistorted image, optionally cropped to remove black borders.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        is_color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Color image
        is_color = True
    else:
        raise ValueError("Unsupported image format. Image must be either grayscale or RGB.")

    # Get image dimensions
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        K, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Undistort the image
    undistorted_image = cv2.undistort(
        image, K, dist_coeffs, None, new_camera_matrix
    )

    # Optionally crop the image to the valid ROI to remove black borders
    x, y, w_roi, h_roi = roi
    undistorted_image = undistorted_image[y:y+h_roi, x:x+w_roi]

    return undistorted_image

def get_images_with_3d_point(self, idx, P_world, image_names = None, tolerance=0.9):            
        c_params = get_camera_params(idx, image_names, self.camera, self.max_seq_len)

        images_names = []
        pixel_coordinate = []
        camera_params_list = []
        
        for key in c_params.keys():
            image_name = key
            c_dict = c_params[key]
            t_cw, R_cw, K = c_dict['t_cw'], c_dict['R_cw'], c_dict['K']
            width, height = c_dict['width'], c_dict['height']
            
            uvs, _ = project_points(P_world, K, R_cw, t_cw)
            
            if uvs is not None:
                u, v = uvs[:, 0]
                w_min, w_max = (0.5 - tolerance/2) * width, (0.5 + tolerance/2) * width
                h_min, h_max = (0.5 - tolerance/2) * height, (0.5 + tolerance/2) * height
                if w_min <= u < w_max and h_min <= v < h_max:
                      
                    images_names.append(image_name)
                    pixel_coordinate.append([u, v])
                    camera_params_list.append(c_dict)
                    
        return images_names, pixel_coordinate, camera_params_list