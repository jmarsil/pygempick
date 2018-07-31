# =============================================================================
# Code for TUTORIAL 5 of pygempick module - Plot PCF's
# =============================================================================

import pygempick.modeling as mod
import pandas as pd

'''
## run & fit the PCF distributions of the data sets...
## scipy.optimize curvefit...
## fit to a poisson & gaussian distributions... 
## what is lambda? - can change r (0,100,dr) - in this case dr in the name 
of the file
## https://en.wikipedia.org/wiki/Complete_spatial_randomness

N = 200 #number of images- no error here 
A = 1018*776
n_cd1 = 1544
n_v30m = 1127

dn = 30 #error rate particles 

lcd1 = (1/N)*n_cd1 #average density of particles lables across this 
test set of images
# r be the number of radius' in question where we could potentially 
find a particle 
# let lambda be the particle density per image...
# see if it follows a poisson distribution...
# missing a factorial?? --- // figure out what that is 

'''
#modified for new fitpcf function!
'''
determine guess filtering parameters
pcfp1 = np.array([100.,1.,1.]) - this is to fit the positive control of the healthy sample
pcfp2 = np.array([10.,1., 1.]) - this is to fit the positive control of the diseased sample

'''
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# the function that I'm going to plot

data1 = pd.read_csv('/home/joseph/Documents/pygempick/supdocs/pcf-dr5-error2.csv',\
                    header=None, skiprows=1)

popt, pcov = mod.fitpcf(data1, 10., 1., 1.)

x = np.arange(0,210,1)
y = np.arange(0,210,1)

spa.pcf(np.arange(0,210,1), popt1[0], popt1[1], popt1[2])


