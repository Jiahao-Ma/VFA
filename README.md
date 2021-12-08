# Voxelized 3D Feature Aggregation for Multiview Detection [[arXiv](https://arxiv.org/abs/2112.03471)]21
 Multiview 3D object detection on MultiviewC dataset through VFA.

## Introduction
We propose a novel method, VFA, for multiview 3D object detection and MultiviewC, a synthetic dataset, for multi-view detection in occlusion scenarios.

## Content
- [MultiviewC dataset](#multiviewc-dataset)
  * [Download MultivewC](#download-multiviewC)
  * [Build your own version](#build-your-own-version)
- [VFA Code](#mvdet-code)
  * [Data Preparation](#data-preparation)
  * [Training and Inference](#training-and-inference)
  * [Evaluation](#evaluation)
## MultiviewC dataset
The MultiviewC dataset mainly contributes to multiview cattle action recognition, 3D objection detection and tracking. We build a novel synthetic dataset MultiviewC through UE4 based on [real cattle video dataset](https://cloudstor.aarnet.edu.au/plus/s/fouvWr9sE6TBueO) which is offered by CISRO.

The MultiviewC dataset is generated on a 37.5 meter by 37.5 meter square field. It contains 7 cameras monitoring cattle activities. The images in MultiviewC are of high resolution, 1280x720 and synthetic animals in our dataset are highly realistic. 

![alt text](https://github.com/Robert-Mar/MultiviewC/blob/main/github_material/MultiviewC.png "Visualization of MultiviewC")

### Download MultiviewC
- download [dataset](#data-preparation) and copy the `annotations`, `images` and `calibrations` folder into this repo. 
### Build your own version
Please refer to this [repo](https://github.com/Robert-Mar/MultiviewC) for MultiviewC dataset toolkits.

## VFA
<img src=https://github.com/Robert-Mar/VFA/blob/main/.github/vfa_v3.png>
This repo is contributed to the code for VFA.

### Data Preparation
In this project, we use [MultiviewC](https://github.com/Robert-Mar/MultiviewC), [MultiviewX](https://github.com/hou-yz/MultiviewX) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/). Download and unzip the dataset in the `~/Data` folder.
Your `~/Data/` folder should look like this
```
Data
├── MultiviewC/
│   └── ...
|
├── MultiviewX/
│   └── ...
|
└── Wildtrack/ 
    └── ...
```

### Training and Inference
Training from scratch.
```
# For MultiviewC
python .\train.py --data MultiviewC

# For MultiviewX
python .\train.py --data MultiviewX

# For Wildtrack
python .\train.py --data Wildtrack
```

We provide the training documents contains the checkpoints of model, optimizer and scheduler and tensorboard containing the training details. Download the latest training documents to `~/experiments` folder from [BaiduDrive](https://pan.baidu.com/s/1KtOBXuxPdnTnKwvyAkZLug)`pwd:6666` or [GoogleDrive](https://drive.google.com/file/d/1SaseZUtc7cb-CX7WoiAQeWe_peqT3yd_/view?usp=sharing) and unzip them. Your `~/experiments/` folder should look like this
```
experiments
├── MultiviewC/
    ├── checkpoints
    |   └── ...
    └── evaluation
    |   └── ...
    └── tensorboard
        └── ...
```

### Evaluation
There are two metrics to evaluate the performance of model. MODA, MODP, Precission and Recall are used to evaluate detection performance such as the detection in occlusion scenes. These metrics need to successfully run in matlab environment. Please refer to [here](https://github.com/Robert-Mar/VFA/tree/main/moft/evaluation) for more details.
Even though, the python implementation of these metrics mentioned above is also provided, it need to select the distance threshould to detemine to positive samples，which is not objective enough. Thus, it is recommended to select the official implementation of matlab.

When it comes to the AP, AOS, OS metrics, we need to install cuda environment and build the toolkit for 3D rotated IoUs calculation. Please refer to this [repo](https://github.com/Robert-Mar/2D-3D-IoUs) for more details.

