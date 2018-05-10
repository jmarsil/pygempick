#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:59:46 2018

@author: joseph

##use the exact same image set but put it throught the picker for various p values...
##draw a normal image with some random noise...
##how effective is this filter
##for varrying p values see how the number of detected circles in the image varies
##can do this for various noise(s)


"""
import cv2
import numpy as np
import pygempick.modeling as mod
import pygempick.core as core
import matplotlib.pyplot as plt


images = 10
n = 250

mu = np.array([.04,.17,.21,.25,.42,.54,.63,.77,.83,.93])
sig = np.array([.013, .08 , .075 ,.1, .175, .22, .25, .27,.34,.43])

detected = []

for j in range(len(mu)):
    
    image, circles, elipses = mod(n, 2, 'yes',images, mu[j],sig[j])
    
    p = np.arange(1,33,2) #ranges of anchor value in filtering kernel
    
    detected_circles = []
    
    for i in range(len(p)): #this is reserved when test 3 or images with varried background conditions are produced...
        
        output = core.igem_filt(p[i],image, 'yes')
    
        keypoints = core.pick(output, 20, .80, .7 , .7, 0)
        
        if len(keypoints) > 0:
                
            detected_circles.append(len(keypoints))
        
        else:
            
            detected_circles.append(0)
        
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im2 = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('/media/joseph/Joe\'s USB/TEST/SEPERATION/{}/test_p{}.jpg'.format(j, p[i]), im2)
    
    detect = np.array(detected_circles)
    
    detected.append(detect)
    
    circ = np.ones(len(detect))*detect[7]
    
    plt.figure()
    plt.title('Separation Power HCLAP Kernel')
    plt.xlabel('Anchor \'p\'')
    plt.ylabel('Detected Particles')
    plt.plot(p,detect, '.b--', label='mu={},sig={}'.format(mu[j],sig[j]))
    plt.plot(p,circ, 'r-', label = 'Drawn')
    plt.legend(loc='best')
    plt.savefig('test_mu{}_sig{}.png'.format(mu[j],sig[j]))

