# Multiview Orthographic Feature Transformation for 3D Object Detection
 Multiview 3D object detection on MultiviewC dataset through moft3d.

## Introduction
We propose a novel method, MOFT3D, for multiview 3D object detection and MultiviewC, a synthetic dataset, for multi-view detection in occlusion scenarios.

## Content
- [MultiviewC dataset](#multiviewc-dataset)
  * [Download MultivewC](#download-multiviewC)
  * [Build your own version](#build-your-own-version)
- [MOFT3D Code](#mvdet-code)
  * [Data Preparation](#data-preparation)
  * [Training Evaluation](#training-evaluation)
  * [Inference](#inference)
## MultiviewC dataset
The MultiviewC dataset mainly contributes to multiview cattle action recognition, 3D objection detection and tracking. We build a novel synthetic dataset MultiviewC through UE4 based on [real cattle video dataset](https://cloudstor.aarnet.edu.au/plus/s/fouvWr9sE6TBueO) which is offered by CISRO.

The MultiviewC dataset is generated on a 37.5 meter by 37.5 meter square field. It contains 7 cameras monitoring cattle activities. The images in MultiviewC are of high resolution, 1280x720 and synthetic animals in our dataset are highly realistic. 

![alt text](https://github.com/Robert-Mar/MultiviewC/blob/main/github_material/MultiviewC.png "Visualization of MultiviewC")

### Download MultiviewC
- download (from [Baidu Drive](https://pan.baidu.com/s/1s67xf8eznms3eF6GfluYSg) `Extraction Code: 6666` and copy the `annotations`, `images` and `calibrations` folder into this repo. 
### Build your own version
Please refer to this [repo](https://github.com/Robert-Mar/MultiviewC) for MultiviewC dataset toolkits.

## MOFT3D
This repo is contributed to the code for MOFT3D.

### Data Preparation
Download the MultiviewC to `~/Data` folder from [BaiduDrive](https://pan.baidu.com/s/1s67xf8eznms3eF6GfluYSg)`pwd:6666` or [GoogleDrive](). And rename it to `MultiviewC_dataset`.

### Training and Inference
Download the latest training documents to `~/experiments` folder from [BaiduDrive](https://pan.baidu.com/s/1OJTZHaDnLh5PJnV7ZqqWmA)`pwd:6666` and unzip them. This training documents contains the checkpoints of model, optimizer and scheduler and tensorboard containing the training details. Notice, this is not the final released version of MOFT3D.

### Evaluation
There are two metrics to evaluate the performance of model. MODA, MODP, Precission and Recall are used to evaluate detection performance such as the detection in occlusion scenes. These metrics need to successfully run in matlab environment. Please refer to [here](https://github.com/Robert-Mar/MOFT3D/tree/main/moft/evaluation) for more details.
Even though, the python implementation of these metrics mentioned above is also provided, it need to select the distance threshould to detemine to positive samples，which is not objective enough. Thus, it is recommended to select the official implementation of matlab.

When it comes to the AP, AOS, OS metrics, we need to install cuda environment and build the toolkit for 3D rotated IoUs calculation. Please refer to this (repo)[https://github.com/Robert-Mar/2D-3D-IoUs] for more details.

