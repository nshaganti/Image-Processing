# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:46:49 2017

@author: nshaganti
"""

""" 

This code detects the different faces in a given picture using the CascadeClassifier.
You may either use webcam to take a new picture or use an existing image.

"""

import cv2
import os
import glob

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
im = cv2.imread("momdad.jpg",0)

cv2.imshow("original", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# Resize the image to speed up detection
mini = cv2.resize(im, (im.shape[1] / size, im.shape[0] / size))

# detect MultiScale / faces 
faces = classifier.detectMultiScale(mini)

ctr = 0
# Draw rectangles around each face
for f in faces:
    (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
    
    #Save just the rectangle faces in SubRecFaces
    sub_face = cv2.resize(im[y:y+h, x:x+w], (250, 250))
    if not os.path.exists('unknownfaces'):
        os.makedirs('unknownfaces')
    FaceFileName = "unknownfaces/face_" + str(ctr) + ".jpg" 
    ctr = ctr + 1
    cv2.imwrite(FaceFileName, sub_face)

del webcam


ctr = 0

for file in glob.glob('C:/Users/nshaganti/Downloads/Personal/upload/*.jpg'):
    im = cv2.imread(file)
    
    mini = cv2.resize(im, (im.shape[1] / size, im.shape[0] / size))

    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v*size for v in f] #Scale the shapesize backup
    
        #Save just the rectangle faces in SubRecFaces
        if ((y-50) > 0 and (x-50) > 0):
            sub_face = cv2.resize(im[y-50:y+h+50, x-50:x+w+50], (250, 250))
        else:
            sub_face = cv2.resize(im[y:y+h, x:x+w], (250, 250))
        if not os.path.exists('unknownfaces'):
            os.makedirs('unknownfaces')
        FaceFileName = "unknownfaces/face_" + str(ctr) + ".jpg"
        ctr = ctr + 1
        cv2.imwrite(FaceFileName, sub_face)
  

# Just a simple script to rename the files in a folder

import os
path = 'C:/Users/nshaganti/Documents/Python Scripts/unknownfaces/New folder'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'Nikhil_Shaganti_' + str(i).zfill(4) +'.jpg'))
    i = i+1
