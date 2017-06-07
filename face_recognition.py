# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:06:02 2017

@author: nshaganti
"""


print __doc__

from time import time
import pylab as pl
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.misc import imread
from scipy.misc import imresize

"""
I'm using the following dataset: The Labeled Faces in the Wild face recognition dataset
This dataset is a collection of JPEG pictures of famous people collected over the internet,
all details are available on the official website: http://vis-www.cs.umass.edu/lfw/.

I have added pictures of my family and myself to this dataset to make it more interesting
after necessary image pre-processing. You can do the same. I've extracted some part of the 
code from stackoverflow 

"""

# I'm only using person's faces for training only if they have 70 or more pictures
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for ML we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print "Total dataset size:"
print "n_samples: %d" % n_samples
print "n_features: %d" % n_features
print "n_classes: %d" % n_classes


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print "Extracting the top %d eigenfaces from %d faces" % (n_components, X.shape[0])
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X)
print "done in %0.3fs" % (time() - t0)

eigenfaces = pca.components_.reshape((n_components, h, w))

print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X)
print "done in %0.3fs" % (time() - t0)

###############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
clf = SVC(kernel='rbf', C = 1000.0, gamma = 0.0005, class_weight='balanced', probability= True)
clf = clf.fit(X_train_pca, y)
print "done in %0.3fs" % (time() - t0)

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_train_pca)
print "done in %0.3fs" % (time() - t0)

print classification_report(y, y_pred, target_names=target_names)
print confusion_matrix(y, y_pred, labels=range(n_classes))


###############################################################################

def preprocess_imgs(img, slice_ = None , color = False, resize = (50,37)):
    
    """Used to preprocess images"""

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    
    default_slice = (slice(0, 250), slice(0, 250))
    
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    # allocate some contiguous memory to host the decoded image slices
    if not color:
        face = np.zeros((1, h, w), dtype=np.float32)
    else:
        face = np.zeros((1, h, w, 3), dtype=np.float32)

    # load the jpeg files as numpy arrays

    face = np.asarray(img[slice_], dtype=np.float32)
    face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
    if resize is not None:
        face = imresize(face, resize)
    if not color:
    # average the color channels to compute a gray levels
    # representaion
        face = face.mean(axis=2)

    return face.reshape(1,-1)
    
# I'm using openCV for the face detection purpose. It uses the Cascade Classifier
# to detect faces. For the face recognition purpose, I'm using the SVM Classifier that I've 
# previously trained.


# Run the code below for face recognition. A webcam will show up and try to recognize
# your face. If your face is not in the training dataset, it will predict as 'not recognized'.
# You can press 'ESC' at anytime to close the webcam window


import cv2

# We load the xml file
classifier = cv2.CascadeClassifier('C:/Users/nshaganti/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0


while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] / size, im.shape[0] / size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        sub_face = cv2.resize(im[y:y+h, x:x+w], (250, 250))
        #Save just the rectangle faces in SubRecFaces
        sub_face = preprocess_imgs(sub_face)
        (prediction, prob) = (target_names[clf.predict(pca.transform(sub_face[[0]]))[0]], clf.predict_proba(pca.transform(sub_face[[0]])).max())
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        if prob > 0.8 :
           cv2.putText(im,'%s' %prediction,(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
           cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
           
 
    # Show the image
    cv2.imshow('Face Recognition', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        cv2.destroyAllWindows()
        del webcam 
        break
