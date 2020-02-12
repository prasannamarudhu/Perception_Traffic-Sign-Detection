import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

from sklearn import svm

import sklearn

hog = cv2.HOGDescriptor()

Descriptors = []
Labels =[]

for i in range(62):

    for filename in glob.glob('Training\Training\{0:05}\*.ppm'.format(i)):

        img = cv2.imread(filename)
        img = cv2.resize(img, (64, 128))

        cv2.moveWindow('',500,90)


        h = hog.compute(img)

        Descriptors.append(h)
        Labels.append(i)


Descriptors = np.resize(Descriptors,(np.shape(Descriptors)[0],np.shape(Descriptors)[1]))
Labels = np.array(Labels).reshape(len(Labels),1)


clf = svm.SVC(kernel='linear', probability=True)

Descriptors = np.array(Descriptors)
data_frame = np.hstack((Descriptors, Labels))
np.random.shuffle(data_frame)


partition = int(len(Descriptors))
x_train= data_frame[:partition,:-1]
y_train = data_frame[:partition,-1:].ravel()

clf.fit(x_train, y_train)

filename = 'Classifier_Data.sav'
pickle.dump(clf, open(filename,'wb'))



