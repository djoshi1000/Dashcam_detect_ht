#This is the main file to run
# Python run.py --config=config/config.yaml
# You can change the directories in the config.yaml file

# Import libraries
import argparse
import yaml
import os, glob
import torch
from src.depth_estimation import DepthEstimator
from src.Process import VideoProcessor
from src.utilities import load_camera_intrinsics, create_output_folder
from ultralytics import YOLO, SAM

def main(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using "{device}" to run the Code.')
    depth_estimator = DepthEstimator(config["depth_model"], device)
    object_detector = YOLO(config["yolo_model"], verbose=False).to(device)
    SAM_object = SAM(config["sam_model"]).to(device)
    camera_intrinsics = load_camera_intrinsics(config)
    output= create_output_folder(config["output_folder"])
    
    # Check if the input is a video or a list of images
    input_path = config["video_path"]  # This should be either a video path or a list of image paths
    
    if os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process as video
        processor = VideoProcessor(input_path, config["output_folder"], config["frame_interval"])
        processor.process_video(depth_estimator, object_detector, SAM_object, camera_intrinsics, config["scale_ratio"])

    elif os.path.isdir(input_path):
        # Process as a list of images in the directory
        image_paths = glob.glob(os.path.join(input_path, "*.png"))  # You can adjust the format if needed
        for image_path in image_paths:
            print(f'Working on {image_path}.')
            processor = VideoProcessor(image_path, output , config["frame_interval"])
            processor.process_images(depth_estimator, object_detector, SAM_object, camera_intrinsics, config["scale_ratio"])

    else:
        raise ValueError("Invalid input format. Provide a valid video file or a directory of image paths.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video/Images Processing Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    main(args.config)
