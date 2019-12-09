#import necessary site_libs
import numpy as np
import math
import os
import glob
import random
import pickle
from imageio import imread
from skimage.transform import resize
class Face_Detect(object):
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
    #firstly, fast calculate haar-like features by integral image
    #Function: integral image
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



    #Function: figure out all the area needed for calculation——horizontal and vertical moving or expanding
    #for simplicity, only take two-rectangle feature into a count,initializing with right and left two rectangle
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
                        # horizontally adjacent
                        if h+2*j<height:
                            top_right=[w,h+j,i,j]
                            feature_areas.append(([top_left],[top_right]))
                        #three_rectangle feature-areas
                        #vertically adjacent
                        if w+3*i<width:
                            bottom_left2=[w+2*i,h,i,j]
                            feature_areas.append(([top_left,bottom_left2],[bottom_left,bottom_left]))
                        # horizontally adajacent
                        if h+3*j<height:
                            top_right2=[w,h+2*j,i,j]
                            feature_areas.append(([top_left, top_right2], [top_right, top_right]))
                        #four_rectangle_feature_areas
                        if w+2*i<width and h+2*j<height:
                            bottom_right=[w+i,h+j,i,j]
                            feature_areas.append(([top_left,bottom_right],[top_right,bottom_left]))
                        j+=1
                    i+=1
        self.feature_areas=feature_areas

    #Function: count the feature value according to feature-areas we have figured out
    #Feature_areas are stores into two parts, that is positive area and negative area
    #each area has the same size [x,y,w,h]
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
        #resize_image = resize(image,(19,19)) #piexl size =(19,19)
        return np.array(image)
    # adaboost algorithm for binary classification
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
            self.test_haar_features=[]
    def cor_rate(self):
        # select top 200 features which have the smaller error
        for i in range(len(self.test_image_features)):
            integeralGraph =self.test_ii[i]
            haarFeatures = self.calHaarFeatures(integeralGraph,self.feature_areas_best)
            #self.test_haar_features.append(haarFeatures)
            self.test_haar_features=np.append(self.test_haar_features,haarFeatures)

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
                if self.test_haar_features[t][i]*self.direction_best>=self.theta_best[t]*self.direction_best:
                    y_hat=1
                else:
                    y_hat=0
                sum_pre+=y_hat*self.alpha[t]
            if sum_pre>=sum_alpha:
                y_pre.append(1)
            else:
                y_pre.append(0)
        # calculate accuracy for test-data
        print(self.test_labels)
        y_pos=y_pre[self.test_labels==1]
        y_neg=y_pre[self.test_labels==0]
        false_neg=(len(y_pos)-sum(y_pos))/len(y_pos)
        false_pos=sum(y_neg)/len(y_pos)
        is_cor=[1 if y_pre[i]==self.test_labels[i] else 0 for i in range(len(self.test_labels))]
        cor_rate_1=sum(is_cor)/len(is_cor)
        return cor_rate_1,false_neg,false_pos
    # extracting data and tranform it into proper form
    # the shape of features should be (J,n)
    # J is the number of haar-like features
    # n is the number of images
    def data_generate(self):
        # only need calculate all haar-like features of training-data
        # for test-data only features of best-feature-areas are needed
        self.getHaarFeaturesArea(self.training_image_features.shape[1],self.training_image_features.shape[2])
#        print(len(self.feature_areas))
        for i in range(len(self.training_image_features)):
            integeralGraph = self.integral_image(self.training_image_features[i])
            haarFeatures = self.calHaarFeatures(integeralGraph,self.feature_areas)
            np.append(haarFeatures,self.training_haar_features)
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
            
            
if __name__ == "__main__":
    fd = Face_Detect()
    fd.load_image()
    fd.data_generate()
   # try:
       # fd.loadVariable(["training_image_features","training_haar_features","test_image_features","test_haar_features","training_labels","test_labels","theta_best","direction_best","alpha","error","best_index","T=T","feature_areas","feature_areas_best","test_ii"])
    #except:
        #fd.data_generate()
        #fd.saveVariable(["training_image_features","training_haar_features","test_image_features","test_haar_features","training_labels","test_labels","theta_best","direction_best","alpha","error","best_index","T=T","feature_areas","feature_areas_best","test_ii"])
    fd.adaboost()


