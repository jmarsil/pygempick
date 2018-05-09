"""
Created on Wed May  9 15:06:15 2018

@author: joseph
"""

import pandas as pd
import numpy as np
import cv2
import scipy.misc as misc

def compress(orig_img):
    '''
    #Takes a large image and compresses it 3.3 times in our case 
    #images are outputed originally as large 9MB images...
    #(That much reolution is unecessary)
    '''
    
    r = 1018/orig_img.shape[1] ##correct aspect ratio of image to prevent distortion
    dim = (1018, int(orig_img.shape[0]*r))
    
    resized_img = cv2.resize(orig_img, dim, interpolation = cv2.INTER_AREA)
    
    return resized_img


##define the filter of the scaled 3x3 laplacian kernel...(High Contract LPF)
def igem_filt(p,image, noise):
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    kernel = np.array([[0,-1,0], [-1,p,-1], [0,-1,0]])
    
    output = cv2.filter2D(gray_img, -1, kernel)
    
    if noise == 'yes':
        
        output = cv2.medianBlur(output,9)
    
    return output

##define difference of Gaussian Filter as used w/ SIFT method...(Lowe,2004)
def dog_filt(tau,image, noise):
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    gaus1 = np.array([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]])
    gaus2 = tau*gaus1 #normally tau = np.sqrt2 , in this case we use as scaling 
                     #factor...
                    
    output1 = cv2.filter2D(gray_img, -1, gaus1)
    output2 = cv2.filter2D(gray_img, -1, gaus2)
    
    if noise == 'yes':

        output1 = cv2.medianBlur(output1,9)
        output2 = cv2.medianBlur(output2,9)
        
    output = output1 - output2
    #returns the filtered binary image...
    return output


def bin_filt(p, image):
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ##If Possible #Histogram Equilization of blurred image
    hist, _ = np.histogram(gray_img.flatten(),256,[0,256])
    #65 to 145 on the x is a good truncation #
    #after this there is a change in concavity...
    ##apending max histogram intensities into a list
    histx = np.argmax(hist)
    
    thresh = histx - np.int(60 + p*1.5)
    ##if there are truly no black dots on this image, this value will be true 
    ##- so set the threshold value to zero...
    
    if np.isnan(thresh) == True:
        thresh = 0
    
    _, thresh_img = cv2.threshold(image, int(thresh), 255, cv2.THRESH_BINARY)
     
    return thresh_img, np.array([thresh, histx])


    
def pick(image, minAREA, minCIRC, minCONV, minINER):
    '''
    #detects immunogold particles on filtered binary image. 
    #have to optimize for each set separately...
    
    '''
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    if minAREA > 0: 
        # Filter by Area.
        params.filterByArea = True
        params.minArea = minAREA
    
    if minCIRC > 0: 
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = minCIRC
    
    if minCONV > 0:
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = minCONV
    
    if minINER > 0:
        #  Filter by Inertia - fix this by theory (what does gold particle look like?)
        params.filterByInertia =True
        params.minInertiaRatio = minINER
    
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;
     
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints1 = detector.detect(image)
    
    return keypoints1


def gamma(a,b,r):
    #function taken from work by Philemonenko et al 2000
    #used as a window covariogram to fink Ripley's K function under w/ boundary
    A = a*b
    return A - (2/np.pi)*(a+b)*r + (1/np.pi)*r**2


def bin2csv(images):
    
    #input is a set of images from folder using glob.glob function...
    #originally from A_correlation_test.py

    i = 0  #image counter
    j = 0
    #------------------------
       
    data = pd.DataFrame()
    #Change picking parameters per test set...
    #p = np.int(input('Center anchor of filter: '))
    minArea = np.int(input('Min Area to Detect: '))
    minCirc = np.float(input('Min Circularity: '))
    minCon = np.float(input('Min Concavity: '))
    minInert = np.float(input('Min Inertial Ratio: '))
    
    for image in images:
        
        orig_img = cv2.imread(image) ##reads specific test file
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        #output = ef.IGEM_filter(p,orig_img, 'no') ##filters image
        keypoints = pick(gray_img, minArea, minCirc, minCon , minInert) ##picks the keypoints
                
        if len(keypoints) > 0:
            #append x and y coordinates of keypoint center pixels
            n = len(keypoints) #number of particles in image
            x = np.zeros(n)
            y = np.zeros(n)
        
                
        k = 0
                
        for keypoint in keypoints:
            ## save the x and y coordinates to a new array...
                    
            x[k] = keypoint.pt[0]
            y[k] = keypoint.pt[1]
                    
            k+=1
                
            df = pd.DataFrame({'x{}'.format(i): x, 'y{}'.format(i) : y})
            data = pd.concat([data,df], ignore_index=True, axis=1)
                
        j += k   
        i +=1
        
    # =============================================================================
    
    ## can use two different plots K at every r for each image on a scatter...
    print('Particles Detected in set : {}'.format(j))
    
    file = input("Name your file")
    
    data.to_csv('{}'.format(file),index=False)
    
    #returns total number of particles detected per binary image in set
    return j

def bin2df(images):
    
    #input is a set of images from folder using glob.glob function...
    #originally from A_correlation_test.py
    
    i = 0  #image counter
    j = 0 #total particles
    #------------------------
       
    data = pd.DataFrame()
    #Change picking parameters per test set...
    #p = np.int(input('Center anchor of filter: '))
    minArea = np.int(input('Min Area to Detect: '))
    minCirc = np.float(input('Min Circularity: '))
    minCon = np.float(input('Min Concavity: '))
    minInert = np.float(input('Min Inertial Ratio: '))
    
    for image in images:
        
        orig_img = cv2.imread(image) ##reads specific test file
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        #output = ef.IGEM_filter(p,orig_img, 'no') ##filters image
        keypoints = pick(gray_img, minArea, minCirc, minCon , minInert) ##picks the keypoints
                
        if len(keypoints) > 0:
            #append x and y coordinates of keypoint center pixels
            n = len(keypoints) #number of particles in image
            x = np.zeros(n)
            y = np.zeros(n)
        
                
        k = 0 #particle counter
                
        for keypoint in keypoints:
            ## save the x and y coordinates to a new array...
                    
            x[k] = keypoint.pt[0]
            y[k] = keypoint.pt[1]
                    
            k+=1
            
             
            
            df = pd.DataFrame({'x{}'.format(i): x, 'y{}'.format(i) : y})
            data = pd.concat([data,df], ignore_index=True, axis=1)
        
        j += k
            
        i+=1

    return data, j #returns data as df and total particles accounted...

def csv2pcf(data, dr):
    
    N = int(input('How Many Processed Images in this set?'))
    ni = int(input('How many particles were detected?')) #nuber of particles counted in test set
    
    ## takes tata from csv produced by bin2csv() and 
    ## outputs non-normalized k and pcf (pair-correlation) spatially invairent 
    ## time series for set of images...
    ## analyzed by bin2csv... taken from A_correlation.py
    
    data1 = pd.read_csv(data, header=None, skiprows=1)
    data = pd.DataFrame(data1)
    
    a = 776 ## number of y pixels in image (height)
    b = 1018 ##number of x pixels in image (width)
    
    #explain why this function is required for statistical detection of immunogold clusters in the data...
    
    l = (1/N)*ni #average density of particles lables across this test set of images
    dni = 50*ni/1000. #false negatives missed by picker per 1000
    
    #correct the boundary effect of image by using the geometric covariogram of the
    # window - defined in Ripley 1988 -- this is given by the gamma function in A_correlation_test
    
    k = []
    dk = []
    pcf = []
    dpcf = []
    
    for r in range(0,100,dr): ##analyze the clustering in a given region between two circles. 
        
        kc = 0
        ki = 0
        
        for p in range(0,len(data),2):
            
            #x,y coordinates of detected immunogold keypoints occur in set of twos in the loaded csv file
            x = np.array(data[p][~np.isnan(data[p])])
            y = np.array(data[p+1][~np.isnan(data[p+1])])
            
            if len(x) > 0:
                
                for i in range(len(x)-1): ##ensure that there are no duplicates while comparing all keypoints
                
                    for j in range(i+1, len(x)):
                            
                        rad = np.sqrt((x[i]-x[j])**2 + (y[i] - y[j])**2)
                            
                        if r < rad <= r+dr: #if radius is less than this area b/w circles
                                
                            kc+= 1/gamma(a,b,rad) #classical K function
                            ki+= 1/(rad*gamma(a,b, rad)) #pair correlation function
                        
                        else:
                            kc+=0   
                            ki+=0
                            
                print(p)
                
        a = kc*(1/(N*l))
        da = a*(dni/ni)
        b = ki*(1/(N*l*2*np.pi))
        db = b*(dni/ni)
       
        k.append(a) #classical K function
        dk.append(da)
        pcf.append(b)
        dpcf.append(db)
        
        print(k,r) 
    
    return k, pcf , dk, dpcf

def keypoints2pcf(data_set, dr):
    
    '''
    Input folder with CSV files of keypoints for different tests
    Need to know Image number and average particles detected in each set
    
    data_set = glob.glob('/home/joseph/Documents/PHY479/Data/anto/*.csv')
    
    output: pcf-dr{}-error.csv - columns dr (sampling radius), pcf 
    (pair correlation coefficient), dpcf (propogated uncertainty in pcf)
    
    '''

    save = pd.DataFrame()
    
    for data in data_set:
        
        print(data)
        
        k, pcf, dk, dpcf = csv2pcf(data, dr)
        
        #plot the Classical K function 
        #Input name of specific data set recorded...
        
        name = input('Test set name:')      
        
        npcf = np.array(pcf)/sum(pcf)
        
        sumd = np.sqrt(sum(np.array(dpcf)**2))
        
        dnpcf = npcf*np.sqrt((np.array(dpcf)/pcf)**2 + (sumd/sum(pcf))**2)
        
        df = pd.DataFrame({'dr-{}'.format(name): np.linspace(1,101, len(pcf)), 'PCF-{}'.format(name): npcf, 'dpcf-{}'.format(name): dnpcf})
        
        save = pd.concat([save,df], ignore_index=False, axis=1)
        
        print(npcf,dnpcf)
    
    save.to_csv('pcf-dr{}-error.csv'.format(dr),index=False)

def pcf(r, N, p0, p1):
    
    return 1/misc.factorial(p1)*(p0**N)*(r**(p1))*np.exp(-p0*r)
