# VJ---Adaboost-Face-Detection_ECEN649
In this project, an implement of Viola-Jones algorithm for face detection is presented. The dataset is from CMU containing both face and non-face figures. There's a training set used for extracting Harr features and build several weak classifiers and then applying these to arrive at a final classifier. The testing set is used for evaluating the accuracy of Viola-Jones algorithm.
## Project Overview
The structure of this algorithm inclueds three parts, extract Haar Features, train to find the best K Haar Features, use Ada-Boost to achieve the detector. 

## Getting started
### Prerequisites 
This implement of VJ face detection requires python version 3.7, and the following packages:

|module|version|usage|
|:--|:--|:--|
numpy|1.16.2|
os||for loading and generating data
glob|0.6|for shuffling data
math||
random||for shuffling data to generate training and testing sets
glob|1.0.0|
imageio|1.0.0|
skimage.transform|1.0.0|

### Usage 
[TOC]

## Create virtual env
    $ python -m venv FaceDetection
    
## Install by using the 'requirements.txt'
    $ pip install -r 'requirements.txt'
    
## Cd to the path and activate
    $ cd FaceDetection/Scripts/
    $ activate.bat
## Run the code



```
Give examples
```


## Evaluation

## Future Work
In this project, the classifers of the face detectors with 1,3,5,10 rounds built and adjusted the threshold. If time is avaliable, (改进算法)

