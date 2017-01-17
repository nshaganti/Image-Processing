# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:46:49 2017

@author: nshaganti
"""

import cv2
import os

size = 4 # this is used later to resize pictures
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:/Users/nshaganti/Downloads/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml')

#read the image here
(rval, im) = webcam.read()
#Flip to act as a mirror
im=cv2.flip(im,1,0) 

#remove the comments for the below code if you wish to use already existing image

"""
im = cv2.imread("nikhil.jpg",0)

cv2.imshow("original", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# Resize the image to speed up detection
mini = cv2.resize(im, (im.shape[1] / size, im.shape[0] / size))

# detect MultiScale / faces 
faces = classifier.detectMultiScale(mini)

# Draw rectangles around each face
for f in faces:
    (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
    cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        #Save just the rectangle faces in SubRecFaces
    sub_face = cv2.resize(im[y:y+h, x:x+w], (250, 250))
    if not os.path.exists('unknownfaces'):
        os.makedirs('unknownfaces')
    FaceFileName = "unknownfaces/face_" + str(y) + ".jpg" 
    cv2.imwrite(FaceFileName, sub_face)

del webcam

