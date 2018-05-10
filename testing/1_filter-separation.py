#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:10:30 2018

@author: joseph
"""
##change parameters of blob detection
##add more thresholds to compare 
##for each test 
import cv2
import time
import glob
import numpy as np
import pygempick.modeling as mod
import matplotlib.pyplot as plt


images = glob.glob('/media/joseph/Joe\'s USB/ALL - CD1/CD1-H14G/Compressed/*.jpg')

#do this for sets of 10 images... 
#combine all the images with values on them into one set... 

N = len(images) #number of images in our set!!!
name = input('Name this image set: ')
i=0
t3=0

#Starting counter for set filters
binary = 0  #binary filter modification as represented in Gold finder...
laplace = 0 #Modified High Contrast Laplacian filter..
gaussian = 0 #Difference of Gaussian Filter...

for image in images[0:100]:
    
    orig_img = cv2.imread(image) ##reads specific test file
    
    p = np.arange(1,31,2) #ranges of anchor value in filtering kernel
    
    t1 = time.time()
    bin1, lap1, dog1 = mod.septest(p, orig_img)
    t2 = time.time()
    t3 += t2-t1 
    
    binary += bin1
    laplace += lap1
    gaussian += dog1
    
    i+=1
    
    if i%10==0:
        print('Done image {} of {}'.format(i,N))


plt.figure()
plt.title('Separation Power HCLAP Kernel - CDI H14G8 ')
plt.xlabel('SCALING FACTOR \'p\'')
plt.ylabel('Detected Particles')
plt.plot(p, binary, '.r-', label='BINARY')
plt.plot(p, laplace ,'.g--', label='HCLAP' )
plt.plot(p,gaussian, '.b--', label='DOG')
plt.legend(loc='best')
plt.savefig('separation_{}.png'.format(name))
plt.grid(True)
print('Average time allapsed of function was {}s'.format(t3/10))

'''
Use the Anti to optimize?
There is variation of picking between methods - finally create one with simple
binary thresholding as a negative control - should be inconsistent as background 
conditions varry...

Set of 10:
draw picked image - lap ring w/green, DOG solid circle in blue
(...control pick with regular binary thresholding...)

#more particles allow for better optimization of the algorithm.
However, in many data sets due to the high resolution and low solute concentrations
in question the # of particles detected varry...

pan specific case - antibody that detects all aTTr forms (normal & diseased)
Cetux - cancer drug (shouldn't see any) - subtract this number as background
H14G1 - specific to misfoolded form of attr. 

One set - (more will come in the future) = need more pan specific 

##see if you can run with similar parameters - how would the result(s) change?

ie 22 .88 .81 .81 for all cetux
ie 20 .83 .73 .73 for all H14G8-mis-ttr
ie 20 .8 .7 .7 for all Anti-poly-ttr

First 100 images of diseased & normal sets were used in the separation tests of
the filter...(We've recived atter )

After the end of the filtering - show the efficiency of picking...across 200 
fabricated with different particles drawn on each... 

Seen similar results when drawn circles vs ellipse(s)...

'''

