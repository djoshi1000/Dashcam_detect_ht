# Dashcam: Detection and height estimation of Objects from Dashcam video 
This repository contains Python implementation of the height estimation of different roadside objects from dashcam video. 


# Introduction
This tool is designed to detect and estimate the heights of trees, poles, and other objects along the roadside from a video captured from the dashboard camera used in the vehicle. Firstly we used You Only Look Once (YOLO) algorithm to train and infer the objects on the video frame. Afterward, we create depth point clouds and images to get real-world information out of the frames. Since the camera uses monocular vision, we utilize the deep learning model of monocular depth estimation (https://github.com/LiheYoung/Depth-Anything) to get the depth information of the objects seen in the image frames of the videos. Various mathematical operations are further done to extract the height information of the detected objects

# Setup
Clone this repo or download the zip file onto your local machine, then install the `requirements.txt` file to install relevant Python packages:

```
$ git clone https://github.com/
$ python install -r requirements.txt
```

# Code Structure
Below is a quick overview of the function of each file.

```bash
########################### height estimation code ###########################    
├──Data                                 #Input image frames
├──Main/
    ├── config/
    │   └── config.yaml                    #Paths for the images or videos should be given here. Carefully analyze required parameters.
    ├── src/
    │   ├── __init__.py
    │   ├── depth_estimation.py    
    │   ├── object_detection.py
    │   ├── point_cloud.py
    │   ├── utilities.py
    │   ├── video_processor.py
    ├── run.py
├──Model                                #best.pt is the preliminary trained yolov8 model with 5 different objects detected
├──Output                                #Output frames saved
├── requirements.txt
├── README.md

```

# Get started
Use `demo.py` to run the code with sample data and default parameters. Execute the following command in the terminal, 
or add `img_path config_fname` in the Parameters when run `demo.py` in the notebook:
```bash
python run.py --config=config/config.yaml
```

The height estimation results will be written to `./data/ht_results/` accordingly.

# How to use
In the detection part, object detection bounding box on the image is used. We then used the Segment Anything Model (SAM) to segment out our object detected inside the bounding box. We used YOLO v8 for this purpose. However, they can be obtained through different neural networks. We have not incorporated the training part of the yolo for object detection. One can manually provide the location of the target object and extract the depth information to furtthe calculate the height of the object. 
  
The installation and usage of the networks can be referred to as their official 
repository. More details about the setup for this study can be referred to the author @ durga.joshi@uconn.eu.
In addition, other networks can be tried for better results. If so, the config file `estimation_config.ini` 
may need to be modified to align with the data, such as the segmentation labels. 



When the above-mentioned result files are prepared, the `demo.py` can be used to estimate heights. 


# Example results

| Fig. 1 | Fig. 2 |
| ---------------------------------- | ----------------------------------  |
| ![fig.1](./output/2024-12-05_18-51-08/frame_1_0.png)              | ![fig.2](./output/2024-12-05_18-51-08/frame_2_0.png)                                     |



# Acknowledgements
We appreciate the open source of the following projects: 

[1] [Depth-anything](https://github.com/LiheYoung/Depth-Anything) \
[2] [Ultralytics](https://github.com/ultralytics/ultralytics.git) 


