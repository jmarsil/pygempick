# =============================================================================
# Code for TUTORIAL 5 of pygempick module - Plot PCF's
# =============================================================================
import pylab as plt
import pygempick.modeling as mod
import pygempick.spatialstats as spa
import numpy as np
import pandas as pd

#modified for new fitpcf function!
'''
determine guess filtering parameters
pcfp1 = np.array([100.,1.,1.]) - this is to fit the positive control of the healthy sample
pcfp2 = np.array([10.,1., 1.]) - this is to fit the positive control of the diseased sample

'''

# the function that I'm going to plot

data1 = pd.read_csv('/home/joseph/Documents/pygempick/supdocs/TEST-pcf-dr5-error2.csv',\
                    header=None, skiprows=1)
# 10., 1., 1.
popt, pcov = mod.fitpcf(data1, 100., 1., 1.)

plt.figure()
x = np.arange(-41,41,.25)
y = np.arange(-41,41,.25)

X,Y = plt.meshgrid(x, y) # grid of point

Z = spa.pcf2d(X,Y, popt[0], popt[1], popt[2])

im = plt.imshow(Z,cmap=plt.cm.coolwarm, extent=(-41, 41, 41, -41)) # drawing the function

#plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
plt.colorbar(im) # adding the colobar on the right
plt.title(r'$P_r= \frac{1}{\beta!}*\lambda^{N}*(\sqrt{x^2 + y^2})^\beta*e^{-\lambda*\sqrt{x^2 + y^2}}$')

plt.show()


