import numpy as np
import cv2

def pixel_to_point(depth_image, camera_intrinsics):
    """
    Converts a depth image to 3D coordinates for each pixel based on camera intrinsics.

    Args:
        depth_image (numpy.ndarray): The depth map (2D array of depth values).
        camera_intrinsics (list): Camera intrinsics in the form [fx, fy, cx, cy].

    Returns:
        tuple: Three numpy arrays representing the X, Y, and Z coordinates of each pixel in 3D space.
    """
    # Get the height and width of the depth image
    height, width = depth_image.shape
    fx, fy, cx, cy = camera_intrinsics
    # Create meshgrid of pixel coordinates (X and Y) for the image
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    # Depth (Z) is directly from the depth image
    Z = depth_image
    # Convert pixel coordinates to 3D space using the camera intrinsics
    X3D = (X - cx) * Z / fx
    Y3D = (Y - cy) * Z / fy
    return X3D, Y3D, Z



def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=1204.0, pixel_x=None, pixel_y=None):
    """
    Generates a point cloud from the depth image and color image, or extracts the 3D coordinates at a specific pixel.

    Args:
        depth_image (numpy.ndarray): The depth map (2D array of depth values).
        color_image (numpy.ndarray): The corresponding color image (2D array of RGB values).
        camera_intrinsics (numpy.ndarray or None): Camera intrinsic matrix (3x3). If None, use the default scale.
        scale_ratio (float): Scale factor to convert depth to meters (default: 1204.0).
        pixel_x (int or None): X-coordinate of the pixel to retrieve 3D coordinates (optional).
        pixel_y (int or None): Y-coordinate of the pixel to retrieve 3D coordinates (optional).

    Returns:
        numpy.ndarray: Depth image (if no pixel_x and pixel_y are given), or depth value at a specific pixel.
    """
    # Get the height and width of the depth image
    height, width = depth_image.shape
    # Resize the color image to match the depth image dimensions
    color_image = cv2.resize(color_image, (width, height))
    # Ensure depth values are not zero (set a minimum value to avoid division by zero)
    depth_image = np.maximum(depth_image, 1e-5)
    # Convert depth values using the scale ratio
    depth_image = scale_ratio / depth_image  # Convert depth to meters (or scaled value)

    if pixel_x is not None and pixel_y is not None:
        depth_value = depth_image[pixel_y, pixel_x]
        if depth_value > 0:
            # Convert 2D pixel coordinates to 3D using camera intrinsics
            X_3D = (pixel_x - camera_intrinsics[0, 2]) * depth_value / camera_intrinsics[0, 0]
            Y_3D = (pixel_y - camera_intrinsics[1, 2]) * depth_value / camera_intrinsics[1, 1]
            Z_3D = depth_value
            # Optionally print the 3D coordinates
            #print(f"3D Coordinates at pixel : X = {X_3D}, Y = {Y_3D}, Z = {Z_3D}")
        else:
            print("Invalid depth value for the specified pixel.")
            
        return depth_value    # Return the depth value at the specific pixel
    return depth_image
