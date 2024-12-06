
# Dashcam: Detection, Height Estimation, and Geolocation of Roadside Objects
This repository contains the Python implementation of a novel approach for detecting and estimating the height of roadside objects using dashcam video. This project is part of an ongoing Ph.D. study exploring real-time roadside vegetation and infrastructure monitoring using in-vehicle data sources. 
### PS: While this repository currently includes code for object detection and height estimation, additional components will be released following the publication of the associated research paper.

# Overview
This tool processes dashboard camera video to detect and estimate the heights of roadside objects, such as trees and poles.

# Methodology:

### Object Detection:
Employs the YOLO v8 algorithm for object detection in video frames.

### Depth Estimation:
Utilizes monocular vision to derive depth information from frames.
Leverages a deep learning model for monocular depth estimation (Depth-Anything) to generate depth point clouds and images.

### Height Calculation:
Integrates depth information with bounding box coordinates of detected objects.
Performs mathematical operations to extract height information of detected objects.

# Installation
Clone the repository and install the required Python packages listed in requirements.txt:

```bash
git clone https://github.com/djoshi1000/Dashcam_detect_ht.git
pip install -r requirements.txt
```

# Structure of the code:
####################### Code #######################    
├── Data/                                 # Input image frames  
├── Main/  
    ├── config/  
    │   └── config.yaml                  # Configuration file for paths and parameters  
    ├── src/  
    │   ├── depth_estimation.py          # Functions for monocular depth estimation  
    │   ├── object_detection.py          # YOLO-based object detection code  
    │   ├── point_cloud.py               # Depth point cloud generation  
    │   ├── utils.py                     # Utility functions for preprocessing and analysis  
    │   ├── Process.py                   # Functions for video frame processing  
    ├── run.py                           # Main execution script  
├── Model/                               # YOLO model weights (e.g., `best.pt`)  
├── Output/                              # Directory for output files (e.g., processed frames)  
├── requirements.txt                     # List of dependencies  
├── README.md                            # Documentation  

# Usage:
```
python run.py --config=config/config.yaml
```

# Configuration
Specify paths and parameters in the config.yaml file:
Video/Input frames path/ output frame paths
YOLO model configuration
Depth estimation parameters

# Note:
Although YOLO v8 is integrated, you can replace it with other object detection algorithms. Update the detection parameters in the configuration file (config.yaml) as needed.
Mathematical operations are applied to calculate height.

### Example Results
Output Frame![input frame](https://github.com/djoshi1000/Dashcam_detect_ht/blob/main/Data/frame_1.png) Output Frame ![outputframe](https://github.com/djoshi1000/Dashcam_detect_ht/blob/main/output/2024-12-05_18-51-08/frame_1_0.png)
	
# Acknowledgments
This project builds upon the open-source contributions of:
[1] [Depth-anything](https://github.com/LiheYoung/Depth-Anything)
[2] [Ultralytics](https://github.com/ultralytics/ultralytics.git) 

For inquiries or further details, contact Durga Joshi at durga.joshi@uconn.edu.
