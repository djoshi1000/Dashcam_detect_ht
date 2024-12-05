import os
import cv2
import numpy as np
from datetime import datetime

def create_output_folder(base_path):
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
    original_height, original_width = original_shape
    new_height, new_width = new_shape
    y_original, x_original = original_pixel

    scale_height = new_height / original_height
    scale_width = new_width / original_width

    new_y = y_original * scale_height
    new_x = x_original * scale_width

    return new_y, new_x

def load_camera_intrinsics(data):

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