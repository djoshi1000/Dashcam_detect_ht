import numpy as np
import cv2

def pixel_to_point(depth_image, camera_intrinsics):
    height, width = depth_image.shape
    fx, fy, cx, cy = camera_intrinsics
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_image
    X3D = (X - cx) * Z / fx
    Y3D = (Y - cy) * Z / fy
    return X3D, Y3D, Z



def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=1204.0, pixel_x=None, pixel_y=None):
    height, width = depth_image.shape
    color_image = cv2.resize(color_image, (width, height))
    
    depth_image = np.maximum(depth_image, 1e-5)
    depth_image = scale_ratio / depth_image  # Convert depth to meters (or scaled value)

    if pixel_x is not None and pixel_y is not None:
        depth_value = depth_image[pixel_y, pixel_x]
        if depth_value > 0:
            # Convert to 3D coordinates using camera intrinsics
            X_3D = (pixel_x - camera_intrinsics[0, 2]) * depth_value / camera_intrinsics[0, 0]
            Y_3D = (pixel_y - camera_intrinsics[1, 2]) * depth_value / camera_intrinsics[1, 1]
            Z_3D = depth_value
            #print(f"3D Coordinates at pixel : X = {X_3D}, Y = {Y_3D}, Z = {Z_3D}")
        else:
            print("Invalid depth value for the specified pixel.")
        return depth_value
    return depth_image