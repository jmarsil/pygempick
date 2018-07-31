#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Code for TUTORIAL 3 of pygempick module - Detected Keypoint Center Recording
# =============================================================================


"""
Created on Fri Oct  6 16:19:36 2017

@author: Joseph Marsilla, MIT

Here we ouline the procedure for plotting the PCF (pair-wise correlation function)
From the cross-correlation function data at given radiu(s) in select image. 

We sum up the probabilitiey for finding particles from r to r+dr and for each 
interval we calculate K at every r in each image. Then we sum up all the K values 
for every image and basically take it's average with the particle density
of the image taken into account. 

For a Homogeneous poisson process - U is a point of X labels that does NOT 
affect the other point. We then compare Kactual vs Ktheroetical if we were seeing
a poisson process equal to (np.pi*r**2). This is the PCF

"""

##change parameters of blob detection
##add more thresholds to compare 
##for each test 

import glob
import pygempick.spatialstats as spa

images = glob.glob('/home/joseph/Desktop/V30M-TEST/*.jpg')

#use these parameters >> keypoints1 = py.pick(output1, 37, .71, .5 , .5, 0) 

N = len(images)
data, blobs = spa.bin2df(images, 2)
file = input("Name your file")
data.to_csv('{}'.format(file),index=False)
print('Number of Images is: {}.'.format(N))
    