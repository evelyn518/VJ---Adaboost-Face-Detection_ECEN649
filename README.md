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
cv2|
 

### Usage 
#### (1). Create virtual env
    
#### (2). Install by using the 'requirements.txt'
    $ pip install -r 'requirements.txt'
  
#### (3). Run the code

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
Fast calculate haar-like features by integral image. Initialize the integral image as zeros with the same size of original image and then iterate.
```
    def integral_image(self,im):
        ii=np.zeros((im.shape[0],im.shape[1]),dtype=np.int32)
        for x in range(im.shape[0]):
            #initialize  the cumulative row sum as zero
            sum_r=0
            for y in range(im.shape[1]):
                sum_r=sum_r+im[x][y]
                ii[x][y]=ii[x-1][y]+sum_r
        return ii
```
Input: the variable list you want to save.
<br>Save the loaded data.

```
    def saveVariable(self,filename):
        """
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
 Figure out all the area needed for calculation——horizontal and vertical moving or expanding
 For simplicity, only take two-rectangle feature into a count,initializing with right and left two rectangle
 ```
     def getHaarFeaturesArea(self,width,height):
        '''
        :param width: original image width
        :param height: original image height
        :return: every sub-windows needed to calculate feature-value
        '''
        #the size-limit of sub-windows
        feature_areas=[]
        for w in range(1,width):
            for h in range(1,height):
                i=0
                while w+i<width:
                    j=0
                    while h+j<height:
                        top_left=[w,h,i,j]
                        #two_rectangle feature-areas
                        #vertically adjacent
                        if w+2*i<width:
                            bottom_left=[w+i,h,i,j]
                            feature_areas.append(([top_left],[bottom_left]))
                            self.areas_label.append('two_rectangle')
                        # horizontally adjacent
                        if h+2*j<height:
                            top_right=[w,h+j,i,j]
                            feature_areas.append(([top_left],[top_right]))
                            self.areas_label.append('two_rectangle')
                        #three_rectangle feature-areas
                        #vertically adjacent
                        if w+3*i<width:
                            bottom_left2=[w+2*i,h,i,j]
                            feature_areas.append(([top_left,bottom_left2],[bottom_left,bottom_left]))
                            self.areas_label.append('three_rectangle')
                        # horizontally adajacent
                        if h+3*j<height:
                            top_right2=[w,h+2*j,i,j]
                            feature_areas.append(([top_left, top_right2], [top_right, top_right]))
                            self.areas_label.append('three_rectangle')
                        #four_rectangle_feature_areas
                        if w+2*i<width and h+2*j<height:
                            bottom_right=[w+i,h+j,i,j]
                            feature_areas.append(([top_left,bottom_right],[top_right,bottom_left]))
                            self.areas_label.append('four_rectangle')
                        j+=1
                    i+=1
        self.feature_areas=feature_areas
```
Function: count the feature value according to feature-areas we have figured out
Feature_areas are stores into two parts, that is positive area and negative area
each area has the same size [x,y,w,h]

```
    def compute_feature(self,ii,feature_area):
        w,h,i,j=feature_area
        return ii[w+i][h+j]+ii[w][h]-ii[w+i][h]-ii[w][h+j]
    def calHaarFeatures(self,ii,feature_areas):
        haarFeatures = []
        for pos_area,neg_area in feature_areas:
            haar_value=sum([self.compute_feature(ii,pos) for pos in pos_area])-sum(
                [self.compute_feature(ii,neg) for neg in neg_area]
            )
            haarFeatures.append(haar_value)
        return haarFeatures
    def load_image(self):
        image_dir=r'C:\Users\gaoru\Desktop\dataset\dataset'
        #training_set
        file_face = os.path.join(image_dir, "trainset", "faces", '*.' + 'png')
        file_non_face = os.path.join(image_dir, "trainset", "non-faces", '*.' + 'png')
        face_list = []
        face_list.extend(glob.glob(file_face))
        non_face_list = []
        non_face_list.extend(glob.glob(file_non_face))
        #test_set
        # prefix t_ represent the test_data
        t_file_face = os.path.join(image_dir, "testset", "faces", '*.' + 'png')
        t_file_non_face = os.path.join(image_dir, "testset", "non-faces", '*.' + 'png')
        t_face_list = []
        t_face_list.extend(glob.glob(t_file_face))
        t_non_face_list = []
        t_non_face_list.extend(glob.glob(t_file_non_face))
        #integrating data
        face_images=np.array([self.read_and_transform(filename) for filename in face_list]) #type as ndarry
        non_face_images=np.array([self.read_and_transform(filename) for filename in non_face_list]) #non-face data
        face_labels=np.ones((len(face_images),)) #labelize face-data
        non_face_labels=np.zeros((len(non_face_images),)) #labelize non-face-data
        # same operation as training_data
        t_face_images=np.array([self.read_and_transform(filename) for filename in t_face_list])
        t_non_face_images=np.array([self.read_and_transform(filename) for filename in t_non_face_list])
        t_face_labels=np.ones((len(t_face_images),))
        t_non_face_labels=np.zeros((len(t_non_face_images),))
        #integrating face and non-face data
        features=np.concatenate((face_images,non_face_images),axis=0)
        labels=np.concatenate((face_labels,non_face_labels),axis=0)
        t_features=np.concatenate((t_face_images,t_non_face_images),axis=0)
        t_labels=np.concatenate((t_face_labels,t_non_face_labels),axis=0)
        #shuffle data
        index=[i for i in range(len(features))]
        random.shuffle(index)
        features=features[index]
        labels=labels[index]
        # color dimention added
        features=np.reshape(features,[-1,19,19,1]) #the 4th-dimention is the color dimention
        t_features=np.reshape(t_features,[-1,19,19,1]) #that the 4th-dimention equals 1 means  images is only grey
        self.training_image_features,self.training_labels,self.test_image_features,self.test_labels=\
            features,labels,t_features,t_labels
    #image into ndarray
    def read_and_transform(self,filename):
        image = imread(filename)
        #resize_image=resize(image)
        return np.array(image)
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

adaboost algorithm for binary classification
```
    def adaboost(self):
        #some important place-parameter
        m=len(self.training_labels[self.training_labels==0]) #number of negative example
        l=len(self.training_labels[self.training_labels==1]) #number of positive example
        n=len(self.training_labels)
        J=len(self.training_haar_features)
        # initialize weights for the first weak classifier
        W = [0.5 / l if self.training_labels[i] == 1 else 0.5 / m for i in range(n)]
        for t in range(self.T):
            # training T weak classifier
            # choose the feature with smallest error as the feature of this weak classifier
            # do J loops
            W = [i / sum(W) for i in W]  # normalize weights
            # compute the total number of positive and negative example
            t_pos = 0
            t_neg = 0
            for _w, _y in zip(W, self.training_labels):
                if _y == 1:
                    t_pos += _w
                else:
                    t_neg += _w
            t_error=np.float('+inf')
            t_direction=None
            t_theta=None
            t_h=None
            t_feature_area=None
            _j = 0  # record the position of the best feature
            for j in range(J):
                # do we need a judgement to filter the features we have picked?
                if j not in self.best_index:
                    f = self.training_haar_features[j]
                    # Each round of boosting selects one feature from the potential features
                    # initialize the error as positive infinite
                    j_error=np.float('+inf')
                    j_direction=None
                    j_theta=None
                    # sort the data by feature value
                    combined_date=sorted(zip(f,W,self.training_labels),key=lambda x:x[0])
                    s_pos=0
                    s_neg=0
                    pos_appear=0
                    neg_appear=0
                    for _f,_w,_y in combined_date:
                        # searching for the proper theta with which the minimunm error is obtained
                        #we cannot say which direction,> or <, is better
                        #so we should put both direction into judgement
                        v_error=min(s_pos+t_neg-s_neg,s_neg+t_pos-s_pos)
                        #judge which direction is better
                        if v_error<j_error: #selecting the mimimum error and record the theta and direction
                            j_error=v_error
                            j_theta=_f
                            j_direction=1 if neg_appear>pos_appear else -1
                        if _y==1:
                            s_pos+=_w
                            pos_appear+=1
                        else:
                            s_neg+=_w
                            neg_appear+=1
                    #obtain the best features with which weak classifier has the minimun error
                    if j_error<t_error:
                        t_error=j_error
                        t_theta=j_theta
                        t_direction=j_direction
                        t_feature_area=self.feature_areas[j]
                        _j=j
                    if j%1000==0:
                        print('loop {}: {:.2f}% features have been fitted'.format(t,100*j/J))
            # calculating beta and alpha in this loop
            t_h=[1 if self.training_haar_features[_j][i]*t_direction>t_theta*t_direction else 0 for i in range(n)]
            print(t_error)
            t_beta=t_error/(1-t_error)
            t_alpha=math.log(1/t_beta)
            # updating weights
            # increasing the weights of which are misclassified
            W=[W[i]*t_beta if t_h[i]!=self.training_labels[i] else W[i] for i in range(n)]
            # put together these parameters
            self.theta_best.append(t_theta)
            self.direction_best.append(t_direction)
            self.alpha.append(t_alpha)
            self.error.append(t_error)
            self.feature_areas_best.append(t_feature_area)
            self.best_index.append(_j)
            print('{}th weak classifier training have completed'.format(t))
            t_a,f_n,f_p=self.cor_rate()
            print('Till now, the total_accurancy of classifier is {:.3f}'.format(t_a))
            print('Till now, the false_neg_rate of classifier is {:.3f}'.format(f_n))
            print('Till now, the false_pos_rate of classifier is {:.3f}'.format(f_p))
            print('Feature number:{}'.format(_j))
            print('Type:{}'.format(self.areas_label[_j]))
            print('Threshold:{}'.format(t_theta))
            print('position:',t_feature_area)
            self.test_haar_features=[]
    def cor_rate(self):
        # select top 200 features which have the smaller error
        for i in range(len(self.test_image_features)):
            integeralGraph =self.test_ii[i]
            haarFeatures = self.calHaarFeatures(integeralGraph,self.feature_areas_best)
            self.test_haar_features.append(haarFeatures)
        self.test_haar_features=np.transpose(self.test_haar_features)
        # set up an empty list to store the y-value of prediction
        y_pre=[]
        # set up threshold
        sum_alpha=0.5*sum(self.alpha)
        # make prediction
        for i in range(len(self.test_labels)):
            # empty variables for prediction-value of each weak classifier
            y_hat = None
            sum_pre = 0
            # make prediction for each test examples
            for t in range(len(self.best_index)):
                if self.test_haar_features[t][i]*self.direction_best>self.theta_best[t]*self.direction_best:
                    y_hat=1
                else:
                    y_hat=0
                sum_pre+=y_hat*self.alpha[t]
            if sum_pre>=sum_alpha:
                y_pre.append(1)
            else:
                y_pre.append(0)
        y_pre=np.array(y_pre)
        # calculate accuracy for test-data
        y_pos=y_pre[np.array(self.test_labels)==1]
        y_neg=y_pre[np.array(self.test_labels)==0]
        false_neg=(len(y_pos)-sum(y_pos))/len(y_pos)
        false_pos=sum(y_neg)/len(y_neg)
        is_cor=[1 if y_pre[i]==self.test_labels[i] else 0 for i in range(len(self.test_labels))]
        cor_rate=sum(is_cor)/len(is_cor)
        return cor_rate,false_neg,false_pos
    # extracting data and tranform it into proper form
    # the shape of features should be (J,n)
    # J is the number of haar-like features
    # n is the number of images
    def data_generate(self):
        # only need calculate all haar-like features of training-data
        # for test-data only features of best-feature-areas are needed
        self.getHaarFeaturesArea(self.training_image_features.shape[1],self.training_image_features.shape[2])
        for i in range(len(self.training_image_features)):
            integeralGraph = self.integral_image(self.training_image_features[i])
            haarFeatures = self.calHaarFeatures(integeralGraph,self.feature_areas)
            self.training_haar_features.append(haarFeatures)
        self.training_haar_features=np.transpose(self.training_haar_features)
        for i in range(len(self.test_image_features)):
            integeralGraph=self.integral_image(self.test_image_features[i])
            self.test_ii.append(integeralGraph)
    # make prediction
    def predict(self,image):
        integeralGraph = self.integral_image(image)
        haarFeatures = self.calHaarFeatures(integeralGraph, self.feature_areas_best)
        sum_alpha = 0.5 * sum(self.alpha)
        sum_pre=0
        for t in range(len(self.best_index)):
            if haarFeatures[t]*self.direction_best>self.theta_best[t]*self.direction_best:
                y_hat=1
            else:
                y_hat=0
            sum_pre += y_hat * self.alpha[t]
        if sum_pre>sum_alpha:
            print('This image is a face-image')
        else:
            print('This image is not a face-image')
```
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
        import cv2
        import os
        
        from os import listdir
        from os.path import isfile, join
        mypath = "C:\\Users\\gaoru\\Documents\\dataset\\figure\\test"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
        
        face_patterns = cv2.CascadeClassifier('C:\\Users\\gaoru\\Documents\\dataset\\figure\\haarcascade_frontalface_default.xml')
        
        for pic in onlyfiles:
        	# print(pic)
        	sample_image = cv2.imread(mypath+"\\" + pic)
        
        	faces = face_patterns.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=5,minSize=(100, 100))
        
        	for (x, y, w, h) in faces:
        	    cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        	cv2.imwrite(mypath +"\\res\\"+ pic + "detect.jpg", sample_image)
```


## Evaluation
### 1. Extract Haar Features

 <b>The total number of Haar Features is: 60426.  
 <b>1.	There are 15390 type 1 (two vertical) features.  
 <b>2.	There are 15390 type 2 (two horizontal) features.  
 <b>3.	There are 10773 type 3 (three horizontal) features.  
 <b>4.	There are 10773 type 4 (three vertical) features.  
 <b>5.	There are 8100 type 5 (four) features.  

### 2. Build Adaboost Detector  
• Feature number :
 <br>Type: Two Horizontal
 <br>Position: (10,3),(9,3)
 <br>Width: 1
 <br>Length: 3
 <br>Threshold: -33
 <br>Training accuracy: 0.801192
 <br>False negative: 0.121328
 <br>False postive: 0.077480

![Feature for 1 round Adaboost](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/round1.png?raw=true)  

• Feature number :
 <br>Type: Two Vertical
 <br>Position: (3,2),(3,6)
 <br>Width: 1
 <br>Length: 4
 <br>Threshold: -112
 <br>Training accuracy: 0.718178
 <br>False negative: 0.115368
 <br>False postive: 0.166454
 
![Feature for 3 rounds Adaboost](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/round2.png?raw=true)  

• Feature number :
 <br>Type: Four
 <br>Position: (8,5),(7,9),(7,5),(8,9)
 <br>Width: 1
 <br>Length: 4
 <br>Threshold: 28
 <br>Training accuracy: 0.756066
 <br>False negative: 0.131545
 <br>False postive: 0.112388
 
![Feature for 5 rounds Adaboost](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/round5.png?raw=true)  

• Feature number :
 <br>Type: Three Horizontal
 <br>Position: (11,2),(14,2),(8,2)
 <br>Width: 3
 <br>Length: 7
 <br>Threshold: -3251
 <br>Training accuracy: 0.756066
 <br>False negative: 0.131545
 <br>False postive: 0.112388
 
![Feature for 10 rounds Adaboost](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/round10.png?raw=true)

```
Computing integral images
Building features
Applying features to training examples
Selecting best features
Selected 5171 potential features
--------------------------------------------------------------------
the 0 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-1019, polarity=1, [RectangleRegion(1, 2, 17, 3)], [RectangleRegion(1, 5, 17, 3)] with accuracy: 0.868221 and alpha: 6.281852
the classifier on 2349 testing data have 1684 acc, 187 nf, 478 pf
the acc rate is 0.716901, false negative is 0.079608, false postive is 0.203491
[(1, 2, 17, 3),(1, 5, 17, 3)]
--------------------------------------------------------------------
the 1 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-608, polarity=-1, [RectangleRegion(0, 0, 12, 5)], [RectangleRegion(0, 5, 12, 5)] with accuracy: 0.267145 and alpha: 6.899264
the classifier on 2349 testing data have 974 acc, 355 nf, 1020 pf
the acc rate is 0.414645, false negative is 0.151128, false postive is 0.434227
[(0, 0, 12, 5),(0, 5, 12, 5)]
--------------------------------------------------------------------
the 2 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-33, polarity=1, [RectangleRegion(10, 3, 1, 3)], [RectangleRegion(9, 3, 1, 3)] with accuracy: 0.823398 and alpha: 8.470672
the classifier on 2349 testing data have 1882 acc, 285 nf, 182 pf
the acc rate is 0.801192, false negative is 0.121328, false postive is 0.077480
[(10, 3, 1, 3),(9, 3, 1, 3)]
--------------------------------------------------------------------
the 3 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-10, polarity=-1, [RectangleRegion(10, 4, 1, 1)], [RectangleRegion(9, 4, 1, 1)] with accuracy: 0.220081 and alpha: 11.440892
the classifier on 2349 testing data have 812 acc, 268 nf, 1269 pf
the acc rate is 0.345679, false negative is 0.114091, false postive is 0.540230
[(10, 4, 1, 1),(9, 4, 1, 1)]
--------------------------------------------------------------------
the 4 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-112, polarity=1, [RectangleRegion(3, 2, 1, 4)], [RectangleRegion(3, 6, 1, 4)] with accuracy: 0.827880 and alpha: 8.805651
the classifier on 2349 testing data have 1687 acc, 271 nf, 391 pf
the acc rate is 0.718178, false negative is 0.115368, false postive is 0.166454
[(3, 2, 1, 4),(3, 6, 1, 4)]
--------------------------------------------------------------------
the 5 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=28, polarity=-1, [RectangleRegion(8, 5, 1, 4), RectangleRegion(7, 9, 1, 4)], [RectangleRegion(7, 5, 1, 4), RectangleRegion(8, 9, 1, 4)] with accuracy: 0.810847 and alpha: 12.408780
the classifier on 2349 testing data have 1776 acc, 309 nf, 264 pf
the acc rate is 0.756066, false negative is 0.131545, false postive is 0.112388
[(8, 5, 1, 4),(7, 9, 1, 4)],[(7, 5, 1, 4), (8, 9, 1, 4)]
--------------------------------------------------------------------
the 6 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-30, polarity=1, [RectangleRegion(10, 3, 1, 3)], [RectangleRegion(9, 3, 1, 3)] with accuracy: 0.814433 and alpha: 8.174293
the classifier on 2349 testing data have 1889 acc, 278 nf, 182 pf
the acc rate is 0.804172, false negative is 0.118348, false postive is 0.077480
[(10, 3, 1, 3),(9, 3, 1, 3)]
--------------------------------------------------------------------
the 7 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-9, polarity=-1, [RectangleRegion(10, 3, 1, 4)], [RectangleRegion(9, 3, 1, 4)] with accuracy: 0.338413 and alpha: 13.159230
the classifier on 2349 testing data have 1619 acc, 320 nf, 410 pf
the acc rate is 0.689229, false negative is 0.136228, false postive is 0.174542
--------------------------------------------------------------------
the 8 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-55, polarity=1, [RectangleRegion(10, 1, 1, 7)], [RectangleRegion(9, 1, 1, 7)] with accuracy: 0.815778 and alpha: 11.626915
the classifier on 2349 testing data have 1805 acc, 327 nf, 217 pf
the acc rate is 0.768412, false negative is 0.139208, false postive is 0.092380
--------------------------------------------------------------------
the 9 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-51, polarity=-1, [RectangleRegion(10, 0, 1, 7)], [RectangleRegion(9, 0, 1, 7)] with accuracy: 0.203048 and alpha: 13.251413
the classifier on 2349 testing data have 1589 acc, 328 nf, 432 pf
the acc rate is 0.676458, false negative is 0.139634, false postive is 0.183908
--------------------------------------------------------------------
the 10 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=151, polarity=-1, [RectangleRegion(7, 0, 1, 9)], [RectangleRegion(6, 0, 1, 9)] with accuracy: 0.824742 and alpha: 10.033712
the classifier on 2349 testing data have 1805 acc, 347 nf, 197 pf
the acc rate is 0.768412, false negative is 0.147722, false postive is 0.083865
--------------------------------------------------------------------
the 11 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=114, polarity=-1, [RectangleRegion(4, 6, 6, 3)], [RectangleRegion(4, 9, 6, 3)] with accuracy: 0.721649 and alpha: 13.737215
the classifier on 2349 testing data have 1811 acc, 284 nf, 254 pf
the acc rate is 0.770966, false negative is 0.120903, false postive is 0.108131
--------------------------------------------------------------------
the 12 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-603, polarity=1, [RectangleRegion(5, 9, 3, 1)], [RectangleRegion(8, 9, 3, 1), RectangleRegion(2, 9, 3, 1)] with accuracy: 0.827432 and alpha: 12.896652
the classifier on 2349 testing data have 1867 acc, 343 nf, 139 pf
the acc rate is 0.794806, false negative is 0.146020, false postive is 0.059174
--------------------------------------------------------------------
the 13 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-3661, polarity=-1, [RectangleRegion(4, 4, 4, 6)], [RectangleRegion(8, 4, 4, 6), RectangleRegion(0, 4, 4, 6)] with accuracy: 0.220977 and alpha: 14.441240
the classifier on 2349 testing data have 1828 acc, 322 nf, 199 pf
the acc rate is 0.778203, false negative is 0.137080, false postive is 0.084717
[(4, 4, 4, 6)],[(8, 4, 4, 6), (0, 4, 4, 6)]
--------------------------------------------------------------------
the 14 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-3896, polarity=1, [RectangleRegion(4, 4, 4, 6)], [RectangleRegion(8, 4, 4, 6), RectangleRegion(0, 4, 4, 6)] with accuracy: 0.802779 and alpha: 14.412803
the classifier on 2349 testing data have 1866 acc, 350 nf, 133 pf
the acc rate is 0.794381, false negative is 0.149000, false postive is 0.056620
--------------------------------------------------------------------
the 15 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-3251, polarity=-1, [RectangleRegion(11, 2, 3, 7)], [RectangleRegion(14, 2, 3, 7), RectangleRegion(8, 2, 3, 7)] with accuracy: 0.223218 and alpha: 14.576724
the classifier on 2349 testing data have 1812 acc, 325 nf, 212 pf
the acc rate is 0.771392, false negative is 0.138357, false postive is 0.090251
[(11, 2, 3, 7)],[(14, 2, 3, 7),(8, 2, 3, 7)]
--------------------------------------------------------------------
the 16 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-2559, polarity=1, [RectangleRegion(11, 5, 3, 5)], [RectangleRegion(14, 5, 3, 5), RectangleRegion(8, 5, 3, 5)] with accuracy: 0.819812 and alpha: 14.456879
the classifier on 2349 testing data have 1864 acc, 356 nf, 129 pf
the acc rate is 0.793529, false negative is 0.151554, false postive is 0.054917
--------------------------------------------------------------------
the 17 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-489, polarity=-1, [RectangleRegion(11, 6, 3, 1)], [RectangleRegion(14, 6, 3, 1), RectangleRegion(8, 6, 3, 1)] with accuracy: 0.219632 and alpha: 14.487232
the classifier on 2349 testing data have 1803 acc, 339 nf, 207 pf
the acc rate is 0.767561, false negative is 0.144317, false postive is 0.088123
--------------------------------------------------------------------
the 18 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-1540, polarity=1, [RectangleRegion(11, 6, 3, 3)], [RectangleRegion(14, 6, 3, 3), RectangleRegion(8, 6, 3, 3)] with accuracy: 0.815778 and alpha: 14.485626
the classifier on 2349 testing data have 1847 acc, 365 nf, 137 pf
the acc rate is 0.786292, false negative is 0.155385, false postive is 0.058323
--------------------------------------------------------------------
the 19 round adboost begin:
Trained 1000 classifiers out of 5171
Trained 2000 classifiers out of 5171
Trained 3000 classifiers out of 5171
Trained 4000 classifiers out of 5171
Trained 5000 classifiers out of 5171
Chose classifier: Weak Clf (threshold=-1006, polarity=-1, [RectangleRegion(11, 5, 3, 2)], [RectangleRegion(14, 5, 3, 2), RectangleRegion(8, 5, 3, 2)] with accuracy: 0.213357 and alpha: 14.482982
the classifier on 2349 testing data have 1790 acc, 340 nf, 219 pf
the acc rate is 0.762026, false negative is 0.144742, false postive is 0.093231
```

### 3. Adjust the threshold  

<b>(1) Choose empirical error as criterion  
 
 ![1](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/Result/emp.png?raw=true)
 
<b>(2) Choose false postive as criterion  
 
 ![1](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/Result/fp.png?raw=true)  
 
<b>(3) Choose false negative as criterion  
 
 ![1](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/Result/fn.png?raw=true)  
 
<b>Table of different criterion  
 
|Criterion|Total Accuracy|False Positive|False Negative|
|:--|:--|:--|:--
Empirical Error|0.80|0.14|0.12
False Positive|0.78|0.00|0.18
False Negative|0.12|0.18|0.17

If empricial error was chosen as criterion, with the increase of number of rounds, the total accuracy will increase and at the end round, it is about 0.80.  

If false positive was chosen as criterion, with the increase of number of rounds, the total accuracy will decrease and at the end round, it is about 0.00.  

If false negative was chosen as criterion, with the increase of number of rounds, the total accuracy will decrease and at the end round, it is about 0.17.  



### 4. Detecting in real-world photos
Orginal photo1:  

![1](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/Barcelona-star-Lionel-Messi-scored-a-hat-trick-against-RCD-Mallorca-1214446.jpg?raw=true)  

After detecting photo1:  

![1](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/after/1.jpg?raw=true)  

Orginal photo2:  

![2](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/MV5BNGE0M2M3NjMtN2ZkZC00NzAyLTgzMWItOGE5ZjFkNTU1NDNjXkEyXkFqcGdeQXVyMzE1MDY2NDg@._V1_.jpg?raw=true)  

After detecting photo1:  

![2](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/after/2.jpg?raw=true)  

Orginal photo3:  

![3](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/maxresdefault.jpg?raw=true)  

After detecting photo1:  

![3](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/after/3.jpg?raw=true)  

Orginal photo4:  

![4](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/images%20for%20vj/La-Liga-Champions_social-image.jpg?raw=true)  

After detecting photo4:  

![4](https://github.com/evelyn518/VJ---Adaboost-Face-Detection_ECEN649/blob/master/after/4.jpg?raw=true)




## Future Work
In this project, the classifers of the face detectors with 1,3,5,10 rounds built and adjusted the threshold. If time is avaliable, I could try to finish building the cascading system.

