#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:55:37 2018

@author: joseph
"""

import cv2
import glob
import pygempick.core as core

#there are 18 particles in this image...
p = 15 #anchor of HCLAP filter...
images = glob.glob('/media/joseph/Joe\'s USB/TEST/picking-test-images/*.jpg')
i = 0


for image in images:
    
    orig_img = cv2.imread(image)

    maskD = core.dog_filt(p,orig_img,'no')
    maskB,_ = core.bin_filt(p,orig_img)
    maskL = core.igem_filt(p,orig_img,'no')
    
    cv2.imwrite('/media/joseph/Joe\'s USB/TEST/picking-test-images/MASK/dog-{}.jpg'.format(i), maskD)
    cv2.imwrite('/media/joseph/Joe\'s USB/TEST/picking-test-images/MASK/bin-{}.jpg'.format(i), maskB)
    cv2.imwrite('/media/joseph/Joe\'s USB/TEST/picking-test-images/MASK/lap-{}.jpg'.format(i), maskL)
    i +=1