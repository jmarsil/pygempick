# =============================================================================
# Code for TUTORIAL 1 of pygempick module - Image Compression
# =============================================================================

import cv2
import glob
import pygempick.core as py


images = glob.glob('/home/joseph/Documents/pygempick/samples/orig/*.tif')

i = 0 #counter

for image in images:
    
    gray_img = py.compress(image)
    
    cv2.imwrite('/media/joseph/Joe\'s USB/Prot-Tech/COMP/nat_samp_{}.jpg'.format(i), gray_img)
    i += 1
    
print('Compression Complete')