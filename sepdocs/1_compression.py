# =============================================================================
# Code for TUTORIAL 1 of pygempick module - Image Compression
# =============================================================================

import cv2
import glob
import pygempick.core as py


images = glob.glob('/media/joseph/Joe\'s USB/ALL - CD1/CD1-H14G/CD1_h14G8_1in200_April212017/*.tif')
histx_max =[] ##contains list of maximum peak of histogram of gray intensities
img_thresholds = []
blob_number = []
image_number = []

i = 0 #counter
p = 11 ##parameter changed w/11
bft = 75##bilateral filter threshold that wil varry - across this set. 
gausk = 3 ##size of gaussian kernal X^D matrix using in pre blurring

for image in images:
    
    orig_img = cv2.imread(image) ##reads specific test file image
    
    gray_img = py.compress(orig_img)
    
    cv2.imwrite('/media/joseph/Joe\'s USB/ALL - CD1/CD1-H14G/Compressed/cd1_h14g8_{}.jpg'.format(i), gray_img)
    i += 1
    
print('Compression Complete')