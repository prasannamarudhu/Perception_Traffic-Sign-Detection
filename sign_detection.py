#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
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


def contrast_norm(channel, minI, maxI):

    iI      = channel 

    minO    = 0

    maxO    = 255

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    
    return iO



def Paste1(Label, x, y, w, h, copy):
    i = [1, 14, 17, 19, 21, 35, 38, 45]

    if Label not in i:
        'ignore'

    else:





        print(x,y,w,h)

        if Label == 1:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 14:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 17:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 19:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 21:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 35:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 38:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image

        elif Label == 45:
            image = cv2.imread('Paste_Img/Img{}.ppm'.format(Label))
            image = cv2.resize(image, (w, h))
            copy[y+h:y+(2*h),x:x+w] = image



def sign_extract(bboxes, regions):
    
    modified_box = list()
    modified_reg = list()
    
    for c, b in enumerate(bboxes):
        
        # calc aspect ratio
        ar = b[2]/b[3]
        
        if ar < 1.5 and ar > 0.6 and b[2] > 20 and b[3]>20:
            
            if len(modified_box) < 1:
                modified_box.append(b)
                modified_reg.append(regions[c])
                
            else:
                for i in range(0, len(modified_box)):
                    
                    if (b[0] > modified_box[i][0] - 40) and (b[0] < modified_box[i][0] + 40):
                        pass
                    else:
                        modified_reg.append(regions[c])
                        modified_box.append(b)
    
    return modified_reg, modified_box

#def PasteImg(y_pred):



for i in range(2139,2861):
    
     
    img = cv2.imread('denoised_input/image{}.jpg'.format(i))

    img = cv2.GaussianBlur(img,(5,5),0)

    img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
    copy = np.copy(img)
    copy2 = np.copy(img)
    
    w = img.shape[0]
    img[int(w*3/5):-1, :] = 0

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    Bc = contrast_norm(B, B.min(), B.max())
    Gc = contrast_norm(G, G.min(), G.max())
    Rc = contrast_norm(R, R.min(), R.max())
    
    # extract the B, G and R from imadjusted values
    zeros = np.zeros_like(Rc)
    
    # intensity normalization     
    
    # for red signs -->
    # Rn = max(0, min(R - B, R - G)/(R + G + B))
    Rn = np.maximum(zeros, np.minimum(Rc - Bc, Rc - Gc)/(Rc + Gc + Bc))
    Rn = np.where(np.isnan(Rn), 0, Rn)
    Rn = 255*2*Rn
    Rn = Rn.astype(np.uint8)
    
    # for blue signs -->
    # Bn = max(0, min(B - R)/(R + G + B))
    Bn = np.maximum(zeros, np.minimum(Bc - Gc, Bc - Rc)/(Rc + Gc + Bc))
    Bn = np.where(np.isnan(Bn), 0, Bn)
    Bn = 255*2.5*Bn
    Bn = Bn.astype(np.uint8)
    Gc = Gc.astype(np.uint8)
    
    # MSER features
    mser1 = cv2.MSER_create(_delta = 13, _min_area = 400, _max_area = 3000)
    regions1, bboxes1 = mser1.detectRegions(Bn)
    regions1, bboxes1 = sign_extract(bboxes1, regions1)
     
    mser2 = cv2.MSER_create(_delta = 13, _min_area = 400, _max_area = 3000)  
    regions2, bboxes2 = mser2.detectRegions(Rn)
    regions2, bboxes2 = sign_extract(bboxes2, regions2)
    
    # extract images of signs to give them to classifier
    if len(bboxes1) is not 0 or len(bboxes2) is not 0: 
        regions = regions1 + regions2
        bboxes = bboxes1 + bboxes2
        hull = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(copy2, hull, 1, (0, 255, 255), 2)

        for j, b in enumerate(bboxes):
            print(b)
            x = b[0]
            y = b[1]
            w = b[2]
            height = b[3]
            roi = copy[y:y+height, x:x+w]
            if len(roi) is not 0:
                #cv2.imwrite("TSR/roi/roi"+str(i)+str(j)+".jpg", roi)

                img = cv2.resize(roi, (64, 128))
                h = hog.compute(img)
                h = h.reshape(1,-1)
                # Labels = np.array(Labels).reshape(len(Labels),1)

                #y_pred = clf.predict(h)

                # y_pred = clf.predict_probs(Descriptors)
                y_pred_proba = clf.predict_proba(h)

                #print("y_pred", y_pred_proba)
                max = np.max(y_pred_proba)
                print("max",max)
                if (max>0.29):
                    _, index = np.where(y_pred_proba == max)
                    print ("index",index)
                    Paste1(index[0],x,y,w,height,copy)
                else:
                    continue


                # ind = y_pred_proba.index(max)


                print("Img number", i)

    cv2.imwrite("Video/Image{}.jpg".format(i),copy)
    cv2.imshow("",copy)
    cv2.waitKey(1)

