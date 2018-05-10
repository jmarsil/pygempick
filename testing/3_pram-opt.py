#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:55:37 2018

@author: joseph
"""

import cv2
import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt
import pygempick.core as core

#there are 18 particles in this image...
p = 21 #anchor of HCLAP filter...
data = pd.DataFrame(columns=['Area', 'Circularity', 'Concavity', 'Inertia', 'Keypoints'])

orig_img = cv2.imread('/media/joseph/Joe\'s USB/TEST/picking-test-images/cd1_h14g8-41.jpg')

mask_img = core.dog_filt(p,orig_img,'no')
t1 = time.time()
count = 0

for i in range(18,36,2):
    for j in np.arange(.7,.92,.02):
        for k in np.arange(.7,.86,.02):
            for s in np.arange(.7,.86,.02):
                
                keypoints = core.pick(mask_img, i, j, k , s) ##picks the keypoints
                data = data.append({'Area': i, 'Circularity': j, 'Concavity':k,'Inertia':s, 'Keypoints':len(keypoints)}, ignore_index=True)
                count +=1
                if count%1000 ==0:
                    print('Count is: ', count)

t2 = time.time()
t3 = t2-t1


index = range(len(data['Keypoints']))
picked = data['Keypoints']
actual = np.ones(len(index))*3


plt.figure()
plt.title('Picking Separation on Anti-CD1 (74) Image')
plt.xlabel('Parameter Combination')
plt.ylabel('Keypoints Detected')
plt.plot(index, picked, '.')
plt.plot(index, actual, 'r-')
plt.show()

file = input("Name your file")    
data.to_csv('{}'.format(file),index=False)