"""
This code reads the input file either video or image frames and first uses the YOLO trained model to detect and track the objects and then uses SAM to segment out the boundary.
The segmented boundary is then used to get the depth value. The depth value is collected from the point cloud created by the depth-anything algorithm point cloud. 
We used 1/10 of the height of the detected object assuming that it would neither be too high nor too low. Then we use simple photogrammetry mathematical logic to acquire the object height. 

The saved frame will have a bounding box around the object, and the Calculated height and the distance from the camera will be displayed. 
"""
import os, glob, cv2
import numpy as np
import gc
from collections import defaultdict
from src.utilities import resize_pixel_location
from src.point_cloud import *

class VideoProcessor:
    def __init__(self, input_path, output_folder, frame_interval=None):
        """
        :param input_path: Path to video or a list of image paths.
        :param output_folder: Folder to save processed frames.
        :param frame_interval: Interval between frames to process (for video only).
        """
        self.input_path = input_path
        self.output_folder = output_folder
        self.frame_interval = frame_interval

    # def process(self, depth_estimator, object_detector, SAM_object, camera_intrinsics, scale_ratio):
        # # Check if the input is a video or a list of images
        # if os.path.isfile(self.input_path) and self.input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # self.process_video(depth_estimator, object_detector, SAM_object, camera_intrinsics, scale_ratio)
        # elif isinstance(self.input_path, list) and all(isinstance(item, str) for item in self.input_path):
            # self.process_images(depth_estimator, object_detector, SAM_object, camera_intrinsics, scale_ratio)
        # else:
            # raise ValueError("Invalid input format. Provide a video file or a list of image paths.")

    def process_video(self, depth_estimator, object_detector, SAM_object, camera_intrinsics, scale_ratio):
        cap = cv2.VideoCapture(self.input_path)
        frame_index = 0
        frame_saved_count = 0
        class_colors = {cls: ((cls * 25) % 255, (cls * 50) % 255, (cls * 75) % 255) for cls in range(5)}  # Random colors for classes
        classNames = object_detector.model.names
        # Initialize tracking variables
        object_ids_dict = defaultdict(int)
        actual_heights_dict = defaultdict(list)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                print("No more frames to read.")
                
            if frame_index % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with the depth model
                frame_depth = depth_estimator.estimate_depth(frame_rgb)

                original_height, original_width, _ = frame.shape
                results = object_detector.track(frame, stream=True, persist=True, tracker="botsort.yaml", verbose=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        cls = int(box.cls[0])
                        conf = round(box.conf[0].item(), 2)

                        # Check if box.id exists before processing it
                        track_ids = int(box.id.int().cpu().numpy().item()) if box.id is not None else -1

                        # Get the class name and assign a unique ID if necessary
                        name = classNames.get(cls)
                        if object_ids_dict[track_ids] == 0:
                            object_ids_dict[track_ids] = max(object_ids_dict.values()) + 1
                        if name == 'Dead':
                            name = 'Tree'

                        object_track_id = f'{name} {object_ids_dict[track_ids]}'
                        bboxes =[(x1, y1, x2, y2)]

                        # Get the segmented mask using SAM
                        segments = SAM_object.predict(frame, bboxes=[(x1, y1, x2, y2)], save=False)
                        masks = segments[0].masks.xy

                        for single_mask in masks:
                            single_mask = np.array(single_mask)

                            # Extract x and y coordinates
                            x_coordinates = single_mask[:, 0]
                            y_coordinates = single_mask[:, 1]

                            min_y = np.min(y_coordinates)
                            max_y = np.max(y_coordinates)
                            base_y = max_y - 30

                            # Find the intersection with the base_y line
                            n = []
                            for i in range(len(single_mask) - 1):
                                pt1, pt2 = tuple(single_mask[i]), tuple(single_mask[i + 1])
                                if min(pt1[1], pt2[1]) <= base_y <= max(pt1[1], pt2[1]):
                                    mx = pt1[0] + (base_y - pt1[1]) * (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
                                    if not np.isnan(mx):
                                        n.append(mx)

                            if len(n) > 2:
                                sorted_n = sorted(list(set(n)))
                                mx = (sorted_n[0] + sorted_n[1]) / 2
                                width = abs(sorted_n[0] - sorted_n[1])
                            elif len(n) == 2:
                                mx = (n[0] + n[1]) / 2
                                width = abs(n[0] - n[1])
                            else:
                                continue

                            # Draw bounding box and mask
                            box_color = class_colors.get(cls, (255, 255, 255))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                            base_y = max_y - int(0.1 * (max_y - min_y))
                            pixels_x, pixels_y = resize_pixel_location(
                                (frame.shape[0], frame.shape[1]),
                                frame_depth.shape,
                                (int(mx), int(base_y))
                            )
                            Dval = create_point_cloud(frame_depth, frame, camera_intrinsics, scale_ratio, int(pixels_x), int(pixels_y))
                            print(f"Depth value at pixel ({pixels_x}, {pixels_y}): {Dval}")
                            if Dval <= 70:
                                h = (max_y - min_y)
                                actual_height = (h / original_width)* 1.78 * Dval
                                actual_heights_dict[object_track_id].append(actual_height)
                                cv2.putText(frame, f"{object_track_id} Ht: {actual_height:.2f} & Dist: {Dval:.2f}",
                                            (x1, y1),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 0, 0), 2)

                frame_filename = os.path.join(self.output_folder, f"frame_{frame_index}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_saved_count += 1
                print(f"Saved frame {frame_index} as {frame_filename}")

            frame_index += 1

        gc.collect()
        cap.release()
        print(f"Total frames saved: {frame_saved_count}")

    def process_images(self, depth_estimator, object_detector, SAM_object, camera_intrinsics, scale_ratio):
        frame_index = 0
        frame_saved_count = 0
        class_colors = {cls: ((cls * 25) % 255, (cls * 50) % 255, (cls * 75) % 255) for cls in range(5)}  # Random colors for classes
        classNames = object_detector.model.names
        # Initialize tracking variables
        object_ids_dict = defaultdict(int)
        actual_heights_dict = defaultdict(list)
        frame = cv2.imread(self.input_path)
        basename= os.path.basename(self.input_path).split('.')[0]
        if frame is None:
            print(f"Failed to read image: {self.input_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the depth model
        frame_depth = depth_estimator.estimate_depth(frame_rgb)

        original_height, original_width, _ = frame.shape
        results = object_detector.track(frame, stream=True, persist=True, tracker="botsort.yaml", verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cls = int(box.cls[0])
                conf = round(box.conf[0].item(), 2)

                # Check if box.id exists before processing it
                track_ids = int(box.id.int().cpu().numpy().item()) if box.id is not None else -1

                name = classNames.get(cls)
                if object_ids_dict[track_ids] == 0:
                    object_ids_dict[track_ids] = max(object_ids_dict.values()) + 1
                if name == 'Dead':
                    name = 'Tree'

                object_track_id = f'{name} {object_ids_dict[track_ids]}'
                bboxes = [(x1, y1, x2, y2)]

                # Get the segmented mask using SAM
                segments = SAM_object.predict(frame, bboxes=[(x1, y1, x2, y2)], save=False)
                masks = segments[0].masks.xy

                for single_mask in masks:
                    single_mask = np.array(single_mask)

                    # Extract x and y coordinates
                    x_coordinates = single_mask[:, 0]
                    y_coordinates = single_mask[:, 1]

                    min_y = np.min(y_coordinates)
                    max_y = np.max(y_coordinates)
                    base_y = max_y - 30

                    # Find the intersection with the base_y line
                    n = []
                    for i in range(len(single_mask) - 1):
                        pt1, pt2 = tuple(single_mask[i]), tuple(single_mask[i + 1])
                        if min(pt1[1], pt2[1]) <= base_y <= max(pt1[1], pt2[1]):
                            mx = pt1[0] + (base_y - pt1[1]) * (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
                            if not np.isnan(mx):
                                n.append(mx)

                    if len(n) > 2:
                        sorted_n = sorted(list(set(n)))
                        mx = (sorted_n[0] + sorted_n[1]) / 2
                        width = abs(sorted_n[0] - sorted_n[1])
                    elif len(n) == 2:
                        mx = (n[0] + n[1]) / 2
                        width = abs(n[0] - n[1])
                    else:
                        continue

                    # Draw bounding box and mask
                    box_color = class_colors.get(cls, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    base_y = max_y - int(0.1 * (max_y - min_y))
                    pixels_x, pixels_y = resize_pixel_location(
                        (frame.shape[0], frame.shape[1]),
                        frame_depth.shape,
                        (int(mx), int(base_y))
                    )
                    Dval = create_point_cloud(frame_depth, frame, camera_intrinsics, scale_ratio, int(pixels_x), int(pixels_y))
                    print(f"Depth value at pixel ({pixels_x}, {pixels_y}): {Dval}")
                    if Dval <= 70:
                        h = (max_y - min_y)
                        actual_height = (h / original_width) * 1.78 * Dval
                        actual_heights_dict[object_track_id].append(actual_height)
                        cv2.putText(frame, f"{object_track_id} Ht: {actual_height:.2f} & Dist: {Dval:.2f}",
                                    (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)

        frame_filename = os.path.join(self.output_folder, f"{basename}_{frame_index}.png")
        cv2.imwrite(frame_filename, frame)
        frame_saved_count += 1
        print(f"Saved frame {frame_index} as {frame_filename}")

        frame_index += 1

        print(f"Total frames saved: {frame_saved_count}")
