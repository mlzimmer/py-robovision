# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:37:13 2018

@author: mlz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = "sm.jpg"
#img = cv2.imread("sm.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.imread(PATH)
cv2.imshow("dog",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#this converts the image to rgb from BGR 
cv_rgb = img[:,:,::-1]


#Grayscale images are easier to process though, so lets make one of those to keep
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_g)
plt.show()

#Use existing cascade
cascPath = "haarcascade_frontalface_default.xml"

#create a haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#Detect faces in the image
#detectMultiScale: General function to detect objects
#1 - image to use: img_g
#2 - scaleFactor: compensates for size difference between images closer to camera
#3 - minNeighbors: how many objects are detected near the current one before it decides its found
#4 - minSize: size of each face window
faces = faceCascade.detectMultiScale(
    img_g,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize = (40, 40),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces".format(len(faces)))


#Draw rectangle around faces
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
 
#Display recognition
cv2.imshow('faces',img)
cv2.waitKey()
cv2.destroyAllWindows()
    
