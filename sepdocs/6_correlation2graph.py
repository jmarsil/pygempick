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


data1 = pd.read_csv('/home/joseph/Documents/PHY479/pcf-dr5-error.csv',\
                    header=None, skiprows=1)

mod.fitpcf(data1)
