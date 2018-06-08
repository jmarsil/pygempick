#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Code for TUTORIAL 2 of pygempick module - Particle Detection
# =============================================================================

#Have a couple of comments - fins a combinatory approach. 
#If image detected is zero, then you run another filter.
#make modification to this case - can be sure to pick the entirity of the oarticles 
#which are burried by depth

#second, use adaptive thresholding (with code provided on github to give filter
#something to work with...this will neutralize the background...)

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pygempick.core as py
#import skimage.feature as feat

detected1 = []
detected2=[]
#detected2 = []
image_number = []

pl = 25 #scaling factor (same scaling factor)
pg = 20 #scaling factor ought to be a multiple of np.sqrt(2)
r = 3.3 #first number that came to mind...
i = 0   #counter for number of image
c = 0   #counter for number of dualy filterd images...

images = glob.glob('/media/joseph/Joe\'s USB/Prot-Tech/COMP/*.jpg')


for image in images:
    
    orig_img = cv2.imread(image) ##reads specific test file
    output1 = py.hlog_filt(pl, orig_img, 'no')
    output2 = py.hlog_filt(pg, orig_img, 'no')
    #output1 = py.dog_filt(pl,orig_img,'no')
    #output2 = py.dog_filt(pg,orig_img, "yes")
    
    #write the image with the picked blobs...
    cv2.imwrite('/media/joseph/Joe\'s USB/Prot-Tech/BIN/nat-hclap23_{}.jpg'.format(i), \
                output1)
    cv2.imwrite('/media/joseph/Joe\'s USB/Prot-Tech/BIN/nat-hlog-18_{}.jpg'.format(i), \
                output2)
     
    keypoints1 = py.pick(output1, 13, .63, .5 , .5, 0) 
    keypoints2 = py.pick(output2, 13, .63, .5 , .5, 0)
    ##picks the keypoints
    #keypoints2 = py.pick(output2, 15, .7, .7 , .7, 0) ##picks the keypoints
    
    keypoints1, dup1 = py.key_filt(keypoints1, keypoints2)
    
    # Draws detected blobs using opencv's draw keypoints 
    imd1 = cv2.drawKeypoints(orig_img, keypoints1, np.array([]), (0,255,0),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imd2 = cv2.drawKeypoints(imd1, keypoints2, np.array([]), (0,255,0),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #imd2 = cv2.drawKeypoints(imd1, keypoints2, np.array([]), (0,0,255),\
    #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #write the image with the picked blobs...
    cv2.imwrite('/media/joseph/Joe\'s USB/Prot-Tech/PICK/picked_{}.jpg'.format(i),\
                imd2)
    
    image_number.append(i)
    i+=1
    
    if len(keypoints1) > 0:
        detected1.append(len(keypoints1))
    else:
        detected1.append(0)
        
    if len(keypoints2) > 0:
        detected2.append(len(keypoints2))
    else:
        detected2.append(0)


print('Total is:', sum(detected1 + detected2))


plt.figure()
plt.title('Aggregate Count in 6E10-Anti images.')
plt.plot(image_number, np.array(detected1) + np.array(detected2), '.b-', label = "Detected w/ LAP")
#plt.plot(image_number, detected2, '.r-', label = "Filtered w/ DOG")
plt.xlabel("Image Number")
plt.ylabel("Particles Detected")
plt.legend(loc='best')
plt.show()


