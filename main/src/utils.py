import os
import cv2
import numpy as np
from datetime import datetime

def create_output_folder(base_path):
    """
    Creates an output folder with a timestamp to store results.

    Args:
        base_path (str): The base directory where the output folder will be created.

    Returns:
        str: The full path to the newly created output folder.
    """
    # Get the current date and time in the desired format (e.g., YYYY-MM-DD_HH-MM-SS)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the full output folder path with the timestamp
    output_folder_with_timestamp = os.path.join(base_path, current_time)
    
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(output_folder_with_timestamp):
        os.makedirs(output_folder_with_timestamp)
        
    return output_folder_with_timestamp

# Function to resize pixel location based on new image dimensions
def resize_pixel_location(original_shape, new_shape, original_pixel):
        """
    Resizes the location of a pixel based on the original image dimensions and new image dimensions.

    Args:
        original_shape (tuple): The original shape of the image (height, width).
        new_shape (tuple): The new shape of the image (height, width).
        original_pixel (tuple): The original pixel location (y, x) in the original image.

    Returns:
        tuple: The resized pixel location (new_y, new_x) in the new image dimensions.
    """
    original_height, original_width = original_shape
    new_height, new_width = new_shape
    y_original, x_original = original_pixel

    scale_height = new_height / original_height
    scale_width = new_width / original_width

    new_y = y_original * scale_height
    new_x = x_original * scale_width

    return new_y, new_x

def load_camera_intrinsics(data):
    """
    Loads the camera intrinsics from a dictionary and returns them as a matrix.

    Args:
        data (dict): A dictionary containing camera intrinsic parameters.

    Returns:
        np.ndarray: A 3x3 camera intrinsics matrix.
    """

    # Extract camera intrinsics
    fx = data['camera_intrinsics']['fx']
    fy = data['camera_intrinsics']['fy']
    cx = data['camera_intrinsics']['cx']
    cy = data['camera_intrinsics']['cy']
    
    # Create the camera intrinsics matrix
    camera_intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return camera_intrinsics
