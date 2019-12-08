# VJ---Adaboost-Face-Detection_ECEN649
In this project, an implement of Viola-Jones algorithm for face detection is presented. The dataset is from CMU containing both face and non-face figures. There's a training set used for extracting Harr features and build several weak classifiers and then applying these to arrive at a final classifier. The testing set is used for evaluating the accuracy of Viola-Jones algorithm.
## Project Overview
The structure of this algorithm inclueds three parts, extract Haar Features, train to find the best K Haar Features, use Ada-Boost to achieve the detector. 

## Getting started
### Prerequisites 
This implement of VJ face detection requires python version 3.7, and the following packages:

|module|version|usage|
|:--|:--|:--|
numpy|1.17.3|for scientific computing, supporting large, multidimensional arrays and matrices
os||for loading and generating data
math||for access to the mathematical functions defined by the C standard
random|1.1.0|for shuffling data to generate training and testing sets
glob|3.5|for finding all the pathnames matching a specified pattern according to the rules used by the Unix shell
imageio|2.6.1|for providing an easy interface to read and write a wide range of image data
skimage.transform||for resizing image to match a certain size 19*19
pickle|2.5|for saving variable, debugging, saving the time of generating and load data
 

### Usage 
#### (1). Create virtual env
    $ python -m venv FaceDetection
    
#### (2). Install by using the 'requirements.txt'
    $ pip install -r 'requirements.txt'
    
#### (3). Cd to the path and activate
    $ cd FaceDetection/Scripts/
    $ activate.bat
#### (4). Run the code

### Project Missions
#### 1. Extract Haar Features
The Haar features are five different kinds of filters with all possible sizes that are applied to the system. Please pay attention
to how they compute the features, use the integral image will save you a massive amount of work. The extracted features
should be a vector with a specific size, carefully calculate how many Haar features you should generate for each picture.
In your report you should include:

The total number of Haar Features is: *****.
1.	There are **** type 1 (two vertical) features.
2.	There are **** type 2 (two horizontal) features.
3.	There are **** type 3 (three horizontal) features.
4.	There are **** type 4 (three vertical) features.
5.	There are **** type 5 (four) features.

Firstly, set global variables
```
    def __init__(self,T=10):
        # set global variables
        self.training_image_features=None
        self.training_haar_features=[]
        self.test_image_features=None
        self.test_haar_features=[]
        self.training_labels=None
        self.test_labels=None
        self.theta_best=[]
        self.direction_best=[]
        self.alpha=[]
        self.error=[]
        self.best_index=[]
        self.T=T  #number of weak classifier
        self.feature_areas=[]
        self.feature_areas_best=[]
        # the integral image of test_examples
        self.test_ii=[]
```
Fast calculate haar-like features by integral image
```
    def integral_image(self,im):
        '''
        :param im:
        :return:intergral image of im
        '''
        # initialize the integral image as zeros
        #with the same size of original image
        ii=np.zeros((im.shape[0],im.shape[1]),dtype=np.int32)
        #iterating
        for x in range(im.shape[0]):
            #initialize  the cumulative row sum as zero
            sum_r=0
            for y in range(im.shape[1]):
                sum_r=sum_r+im[x][y]
                ii[x][y]=ii[x-1][y]+sum_r
        return ii
```

```
    def saveVariable(self,filename):
        """
        This function is going to save the loaded data
        Input is the variable list you want to save 

        with open('original'+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.data, f)
        f.close()
        """
        if filename:
            for file in filename:
                with open(file+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                    exec(source='pickle.dump('+ 'self.'+ file + ', f)')
                f.close()
    def loadVariable(self,filename):
        if filename:
            for file in filename:
                try:
                    with open(file+'.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
                        # eval('self.' + filename + '=pickle.load(f)')
                        exec(source='self.' + file + '=pickle.load(f)')
                    f.close()
                except:
                    raise Exception()
```
#### 2. Build Adaboost Detector
After extracting the features, employ the AdaBoost algorithm and find the detector with 1, 3, 5, 10 rounds. For each different
detector, you need to show the feature you choose, the threshold you have, and at last, draw your top one feature for each
detector on a test image (like the original paper).
In your report, you need to include:

• Feature number 1:
 <br>Type: Two Horizontal
 <br>Position: (10,10)
 <br>Width: 8
 <br>Length: 4
 <br>Threshold: 0.5555
 <br>Training accuracy: 0.61

#### 3. Adjust the threshold
In the real world, there are different standards for the face detection system. We may want to eliminate as much false alarm
as possible in daily life, for example, we can tolerate not being recognized as a face but cannot tolerate the whole environment
are all recognized as faces. However, if you build the system for security reasons, you don’t want to miss any of the suspicious
moves.
In order to balance these two kinds of losses, we need to use different criteria to train your detector, for example, use false
positive instead of the empirical error. Your mission is to train different 5 round AdaBoost detectors, show us how their false
positive and false negative changes. What change did you make on your system? 

#### 4.Build the cascading system
Use the cascading method in our class, train the detector up to 40 rounds, and compare it with the single AdaBoost detector.
Does your result get better? Does the training get faster? Compare your results with what you have got in our baseline part.
In your report, you need to show me after passing each detector in your cascading system, how many non-faces photos
have you abandoned? What is the final accuracy of this cascaded system? Can you explain why this improvement/fallback
happens?


```
Give examples
```


## Evaluation
### 1. Extract Haar Features

### 2. Build Adaboost Detector

### 3. Adjust the threshold
### 4. Build the cascading system

## Future Work
In this project, the classifers of the face detectors with 1,3,5,10 rounds built and adjusted the threshold. If time is avaliable, (改进算法)

