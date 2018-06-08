# =============================================================================
# Code for TUTORIAL 4 of pygempick module - Calculate Correlation, PCF w/ Error
# =============================================================================

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygempick.spatialstats as spa


'''
Taken From 2000, Philmonenko:

    The value of K ^-1 (r 1 , r 2 ) has an
    easy interpretation: we can detect deviations from
    randomness by comparing it directly to the differ-
    ence of the diameters r 2 and r 1 (r 2 ⫺ r 1 ) because the
    mean value of K ^-1 (r 1 , r 2 ) for completely random
    point pattern is equal to r 2 ⫺ r 1
    
    We recommend the use of the histogram representa-
    tion (of the PCF) as it demonstrates the characteristics of clus-
    tering and/or colocalization (characteristic distance 
    and degree of clustering or colocalization) of gold
    particles in a clear and comprehensive manner. The
    confidence intervals of heights of the histogram bars
    allow the investigator to judge whether the devia-
    tions from randomness are statistically significant.
    
'''
##
data_set = glob.glob('/home/joseph/Documents/PHY479/Data/anto/*.csv')
#data_set = glob.glob('/home/joseph/Documents/V30M-Keypoints/*.csv')
dr = 5

save = pd.DataFrame()

for data in data_set: 
    #if you have multiple csv folders that you would like CSV data calculated 
    #for, program was made to track data from more than one CSV folder
    
    
    print(data)
    
    k, pcf, dk, dpcf = spa.csv2pcf(data, dr)
    
    #plot the Classical K function 
    #Input name of specific data set recorded...
    
    name = input('Test set name:')      
    
    if len(k)>0:
        
        stdk = np.sqrt(sum(k/sum(k))/len(k))
        
        plt.figure(1)
        plt.title('K Function of V30M Micrographs')
        
        #plot at functions midpoint...
        plt.plot(np.linspace(1,101,len(k)), k/sum(k), label='Classical K function for {}'.format(name))
        #if len(k) > 2:
        #    plt.errorbar(np.arange(1,101,2), k/sum(k) , yerr=stdk, fmt='o')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        
        stdpcf = np.sqrt(sum(pcf/sum(pcf))/len(pcf))
    
        plt.figure(2)
        plt.title('PCF of V30M Micrographs')
        plt.plot(np.linspace(1,101, len(pcf)), pcf/sum(pcf), label='Pair Correlation Function {}'.format(name))
        #if len(k) > 2:
        #    plt.errorbar(np.arange(1,101,2), pcf/sum(pcf) , yerr=stdpcf, fmt='o')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    npcf = np.array(pcf)/sum(pcf)
    
    sumd = np.sqrt(sum(np.array(dpcf)**2))
    
    dnpcf = npcf*np.sqrt((np.array(dpcf)/pcf)**2 + (sumd/sum(pcf))**2)
    
    df = pd.DataFrame({'dr-{}'.format(name): np.linspace(1,101, len(pcf)), 'PCF-{}'.format(name): npcf, 'dpcf-{}'.format(name): dnpcf})
    
    save = pd.concat([save,df], ignore_index=False, axis=1)
    
    print(npcf,dnpcf)

save.to_csv('pcf-dr{}-error.csv'.format(dr),index=False)

    