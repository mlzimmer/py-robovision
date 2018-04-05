# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:52:53 2018

@author: mlz
"""

import cv2
import sys

#designating a classifier
CSPTH = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(CSPTH)

#video feed
l_vid = cv2.VideoCapture(0)

while True:
    ret, frame = l_vid.read()
    #easier to process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize = (30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #Draw around face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('video',frame)
    
    #wait for 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
l_vid.release()
cv2.destroyAllWindows()