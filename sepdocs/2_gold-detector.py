#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Code for TUTORIAL 2 of pygempick module - Particle Detection
# =============================================================================

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pygempick.core as py

detected1 = []
detected2 = []
image_number = []

pl = 15  #scaling factor (same scaling factor)
pg = 4.79 #scaling factor ought to be a multiple of np.sqrt(2)
r = 3.3 #first number that came to mind...
i = 0   #counter
images = glob.glob('/media/joseph/Joe\'s USB/DOG1/*.jpg')

for image in images:
    
    orig_img = cv2.imread(image) ##reads specific test file
    output1 = py.igem_filt(pl,orig_img,"yes")
    output2 = py.dog_filt(pg,orig_img, "yes")
    
    keypoints1 = py.pick(output1, 15, .7, .7 , .7, 0) ##picks the keypoints
    keypoints2 = py.pick(output2, 15, .7, .7 , .7, 0) ##picks the keypoints
    
    # Draws detected blobs using opencv's draw keypoints 
    imd1 = cv2.drawKeypoints(orig_img, keypoints1, np.array([]), (255,0,0),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imd2 = cv2.drawKeypoints(imd1, keypoints2, np.array([]), (0,0,255),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #write the image with the picked blobs...
    cv2.imwrite('/media/joseph/Joe\'s USB/DOG1/Picked/picked_{}.jpg'.format(i),\
                imd1)
    
    image_number.append(i)
    i+=1
    
    if len(keypoints1) > 0:
                
        detected1.append(len(keypoints1))
        detected2.append(len(keypoints2))
    else:
            
        detected1.append(0)
        detected2.append(0)


plt.figure()

plt.plot(image_number, detected1, '.b-', label = "Detected w/ LAP")
plt.plot(image_number, detected2, '.r-', label = "Filtered w/ DOG")

plt.xlabel("Image Number")
plt.ylabel("Particles Detected")
plt.legend(loc='best')

plt.show()


