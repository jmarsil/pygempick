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

'''
Note - this is an over representation of the error rate. 
(Images in both sets had varrying background conditions)

Counts include background and false positive(s) - will be used as error.

Blank Test 1: Total 49 particles detected in 75 images with 7 duplicates. (Lots of false positives)
(If that were the case 154 missed in test 1 with same conditions...) 

Blank Test 2: Total 45 particles detected in 75 images with 1 duplicate. (correlates to 141 missed
in 235 imges)

Blank Test 3: Total 33 particles detected in 75 images with 7 duplicates (correlates to 104 missed
in 235 images)

Blank Test 4: Total 18 particles detected in 75 images with 3 duplicates (correlates to 56 particles 
missed in 235 images)

Blank Test 5: Total (Written down in book - record here)

Test 7 (tri) - 2502 particles detected with 815 duplicates. t8 - 2457, 739
'''

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pygempick.core as py
#import skimage.feature as feat

detected1 = []
detected2 = []
detected3 = []
image_number = []

p1 = 18 #scaling factor (same scaling factor)
p2 = 23 #scaling factor ought to be a multiple of np.sqrt(2)
i = 0   #counter for number of image
c = 0   #counter for number of dualy filterd images...

duplicates = 0

images = glob.glob('/media/joseph/Joe\'s USB/05.08.2018/Joe Trial/6E10_compressed/*.jpg')

for image in images:
    
    orig_img = cv2.imread(image) ##reads specific test file
    output1 = py.hclap_filt(p2, orig_img, 'no')
    output2 = py.hlog_filt(p1, orig_img, 'no')
    output3 = py.hlog_filt(p2, orig_img, 'no')
    #output1 = py.dog_filt(pl,orig_img,'no')
    #output2 = py.dog_filt(pg,orig_img, "yes")
    
    #write the image with the picked blobs...
    #cv2.imwrite('/media/joseph/Joe\'s USB/05.08.2018/Joe Trial/6E10_Binary/HCLAP-18/6E10-hclap_{}.jpg'.format(i), \
     #           output3)
    #cv2.imwrite('/media/joseph/Joe\'s USB/05.08.2018/Joe Trial/6E10_Binary/HLOG-23/R10/R10-hlog_{}.jpg'.format(i), \
                #output2)
    
    keypoints1 = py.pick(output1, 31, .83, .5, .5, 0) 
    keypoints2 = py.pick(output2, 31, .83, .5, .5, 0)
    keypoints3 = py.pick(output3, 31, .83, .5, .5, 0)
    
    keypoints1, dup1 = py.key_filt(keypoints1, keypoints2)
    
    keypoints2, dup2 = py.key_filt(keypoints2, keypoints3)
    
    keypoints3, dup3 = py.key_filt(keypoints3, keypoints1)
    
    duplicates += dup1 + dup2 + dup3 #duplicates between three filters
    
    # Draws detected blobs using opencv's draw keypoints 
    imd1 = cv2.drawKeypoints(orig_img, keypoints1, np.array([]), (0,0,255),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imd2 = cv2.drawKeypoints(imd1, keypoints2, np.array([]), (0,255,0),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imd3 = cv2.drawKeypoints(imd2, keypoints3, np.array([]), (0,255,0),\
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #imd2 = cv2.drawKeypoints(imd1, keypoints2, np.array([]), (0,0,255),\
    #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #write the image with the picked blobs...
    cv2.imwrite('/media/joseph/Joe\'s USB/05.08.2018/Joe Trial/6E10_picked/Test8/6E10-picked_{}.jpg'.format(i),\
                imd3)
    
    image_number.append(i)
    i+=1
    
    if i%10 == 0:
        print(i, ' Number of duplicates:', duplicates)
    
    if len(keypoints1) > 0:
        detected1.append(len(keypoints1))
    else:
        detected1.append(0)
        
    if len(keypoints2) > 0:
        detected2.append(len(keypoints2))
    else:
        detected2.append(0)
        
    if len(keypoints3) > 0:
        detected3.append(len(keypoints3))
    else:
        detected3.append(0)

print('Total is:', sum(detected1 + detected2 + detected3))

plt.figure()
plt.title('AB Aggregate Count in 6E10-Anti images.')
plt.plot(image_number, np.array(detected1) +np.array(detected2)+ np.array(detected3), '.b-', label = "HCLAP-23 & HLOG-18 & HLOG-23")
#plt.plot(image_number, detected2, '.r-', label = "Filtered w/ DOG")
plt.xlabel("Image Number")
plt.ylabel("Particles Detected")
plt.legend(loc='best')
plt.show()


