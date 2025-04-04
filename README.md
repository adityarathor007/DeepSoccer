# ⚽ DeepSoccer: A Deep Learning Framework for Football Video Analysis

In this repo we designed a framework to analyze football videos using YOLO for real-time object detection and tracking. Beyond simple detection, it integrates 2D spatial projections, area matching, and advanced movement analysis to better understand player and ball dynamics.

This approach enables more accurate scene interpretation and supports applications such as game analytics, performance insights, and tactical evaluation. Our experimental results show that DeepSoccer can effectively track multiple entities on the field and extract meaningful data from match footage.



## Dataset Download

training samples can be download from the following link along with chossing the yolov11:
- For ball,players,gk,referree detection: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/14

- For pitch keypoints detection https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi

for testing samples 

```bash
    cd data
    gdown "https://drive.google.com/uc?id=1-CnEfapV2sjA8wM_gxTLMdHGwFuxEjbT"
    unzip tests
    
```

you can also download samples from the Kaggle - [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) 


## Directory Structure

    .
    ├── dataset
    |   ├── roboflow        #for storing dataset used for fine tuning detection model for all 4 classes
    |   ├── roboflow_ball   #for storing dataset used for fine tuning detection model for all 4 classes
    |   ├── roboflow_pitch  #for storing dataset used for fine tuning pose detection yolo model              
    |   └── tests           #for storing samples to be tested and results obtained
    ├── runs                #to store fine-tuned model 
    │   ├── train          
    |   └── train_ball              
    ├── utils
    |    ├── field_config.py
    |    ├── field_annotator.py
    |    └── team.py
    ├── 2D_Projections.ipynb
    ├── README.md
    ├── analyze.ipynb
    ├── demo.mp4
    ├── img_detections.ipynb
    ├── pitch_keypoint_detection.ipynb
    ├── team_clustering.ipynb
    ├── train.ipynb
    └── visualize.ipynb


    
## Results

Some of the results can be downloaded from this link
``` bash 
gdown "https://drive.google.com/uc?id=16xDEt4gAMq9o_xEd9iTbrfHbE2pHGE_k"

``` 