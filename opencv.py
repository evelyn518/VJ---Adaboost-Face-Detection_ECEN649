# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:16:03 2019

@author: gaoru
"""
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
