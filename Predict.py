import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


import sklearn

hog = cv2.HOGDescriptor()

Descriptors = []
Labels=[]

filename = open("Classifier_Data.sav",'rb')
clf = pickle.load(filename)


for i in range(62):

    for filename in glob.glob('Testing\Testing\{0:05}\*.ppm'.format(i)):

        img = cv2.imread(filename)
        img = cv2.resize(img, (64, 128))

        h = hog.compute(img)


        Descriptors.append(h)
        Labels.append(i)


Descriptors = np.resize(Descriptors,(np.shape(Descriptors)[0],np.shape(Descriptors)[1]))
Labels = np.array(Labels).reshape(len(Labels),1)

y_pred = clf.predict(Descriptors)


print("Accuracy: "+str(accuracy_score(Labels, y_pred)))
print('\n')
print(classification_report(Labels, y_pred))
