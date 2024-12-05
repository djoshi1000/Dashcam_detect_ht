# Dashcam: Detection and height estimation of Objects from Dashcam video 
This repository contains Python implementation of the height estimation of different roadside objects from dashcam video. 


# Introduction
This tool is designed to detect and estimate heights of trees, poles and other objects along the roadside from a videos captured from dashboard camera used in the vehicle. Firstly we used You Only Look Once (YOLO) algorithm to train and inference the objects on the video frame. Afterwards, we create depth point clouds and image to get the real world information out of the frames. Since the camera uses monocular vision, we utilize the deep learning model of monocular depth estimation (https://github.com/LiheYoung/Depth-Anything) to get the depth information of the objects seen in the image frames of the videos. Various mathematical operations are further done to extract the height information of the detected objects

# Setup
Clone this repo or download the zip file onto your local machine, then install `requirements.txt` file to install relevant python packages:

```
$ git clone https://github.com/
$ python install -r requirements.txt
```

# Code Structure
Below is a quick overview of the function of each file.

```bash
########################### height estimation code ###########################    
config/                         # configurations
    estimation_config.ini       # default parameters for height estimation
data/                           # default folder for placing the data
    imgs/                       # folder for image frame extracted from video
misc/                           # misc files
demo.py                         # main function
filesIO.py                      # functions for creating image frames from video files
heightMeasurement.py            # functions for height measurement
```

# Get started
Use `demo.py` to run the code with sample data and default parameters. Execute the following command in the terminal, 
or add `img_path config_fname` in the Parameters when run `demo.py` in the notebook:
```bash
python ./demo.py ./data/imgs/ ./config/estimation_config.ini
```

The height estimation results will be written to `./data/ht_results/` accordingly.

# How to use
In the detection part, object detection bounding box on the image is used. We then used Segment Anything Model (SAM) to segment out our object detected inside the bounding box. We used YOLO v8 for this purpose. However, they can be obtained through different neural networks. We have not incorporated the training part of the yolo for object detection. One can manually provide the location of the target object and extract the depth information to furtthe calculate the height of the object. 
  
The installation and usage of the networks can be referred to their official 
repository. More details about the setup for this study can be referred to the author @ durga.joshi@uconn.eu.
In addition, other networks can be tried for better results. If so, the config file `estimation_config.ini` 
may need to be modified to align with the data, such as the segmentation labels. 



When the above mentioned result files are prepared, the `demo.py` can be used to estimate heights. 


# Example results

| Fig. 1 | Fig. 2 |
| ---------------------------------- | ----------------------------------  |
| ![fig.1](./misc/figs/0001.png)              | ![fig.2](./misc/figs/0002.png)                                     |



# Acknowledgements
We appreciate the open source of the following projects: 

[1] [Depth-anything](https://github.com/LiheYoung/Depth-Anything) \
[2] [Ultralytics]() 


