"""
Created on Wed May  9 15:06:15 2018

@author: joseph
"""

import pandas as pd
import numpy as np
import cv2
import random
import scipy.misc as misc
import scipy.optimize as opt
import matplotlib.pyplot as plt

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
def IGEM_filter(p,image, noise):
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    kernel = np.array([[0,-1,0], [-1,p,-1], [0,-1,0]])
    
    output = cv2.filter2D(gray_img, -1, kernel)
    
    if noise == 'yes':
        
        output = cv2.medianBlur(output,9)
    
    return output

##define difference of Gaussian Filter as used w/ SIFT method...(Lowe,2004)
def DOG_filter(tau,image, noise):
    
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


def BIN_filter(p, image):
    
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


    
def IGEM_pick(image, minAREA, minCIRC, minCONV, minINER):
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
    
    
def IGEM_draw(n, test_number, noise, images):
    
    '''
    n = particle number of real data set to make fake distribution... always same
    
    test_number: 1 for only circles , 2 for both circles and ellipses...
    '''
    
    row = 776  #image height
    col = 1018 #image width
    
    radrange = np.arange(4,8,1)
    
    
    mu = n/images #mean particle number across your images
    sigma = np.sqrt(mu) #standard deviation of the mean from your data
    
    ##creates a new normal distribution based on your data (particles,images)
    pick = np.random.normal(mu,sigma)
    
    #height = np.arange(26,750) ##array of possible particle heights
    #width = np.arange(26,992)   ##array of possible particle widths
    height = 750
    width =  990
    count = 0
    circles = 0
    elipses = 0
    #mu1 = .05
    #sig1 = .02
    
    image = 255*np.ones((row,col), np.float32)
    ##convert to BGR
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if noise == 'yes':
        
        mu1 = input('Input mean of Gaussian Distributed Noise')
        sig1 = input('Input std of Gaussian Distributed Noise')
        
        ##adding random gaussian distributed noise to image...
        for q in range(row):
            for w in range(col):
                
                image[q][w] = np.float32(np.int(255*np.random.normal(mu1,sig1))) 
                ##change this value for high variability in background conditions..
        
    if test_number == 1:
        
        for j in range(np.int(pick)):
            
            count+=1
            
            ##picks a random particle radius between 4 and 8 pixels
            r = random.choice(radrange)
            
            ##chooses a random center position for the circle
            #h = random.choice(height)
            #w = random.choice(width)
            
            w = np.random.uniform(20,width)
            h = np.random.uniform(20,height)
            
            #w = np.int(col*np.random.rand()) #first method used to choose random width/height...
            
            ##ensure that no particles are drawn on the edges of the image
            ##figure out how to void borders...
                
            ##draw a black circle
            cv2.circle(image,(h,w), np.int(r), (0,0,0), -1)
        
        image = (image).astype('uint8')
        print('Complete')
        return image, count

    elif test_number == 2:
        
        q = np.int(pick)
        count = 0
        
        while count <= q:
            
            ##picks a random particle radius between 4 and 8 pixels
            axis = random.choice(radrange)
            #N = width * height / 4
            ##chooses a random center position for the circle
            w = np.int(np.random.uniform(20,width))
            h = np.int(np.random.uniform(20,height))
            
            
              ##bernouli trial to draw either circle or elippse...
            flip = np.random.rand()
            
            if flip < 0.5:
                #draw a circle
                cv2.circle(image,(h,w), np.int(axis), (0,0,0), -1)
                circles +=1
            
            else:
                #draw an elippse...
                elipses += 1
                cv2.ellipse(image,(h,w),(int(axis)*2,int(axis)),0,0,360,(0,0,0),-1)
            
            count += 1
        
        
        count = circles + elipses
        image = (image).astype('uint8')
        return image, int(circles), int(elipses)
    


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
        keypoints = IGEM_pick(gray_img, minArea, minCirc, minCon , minInert) ##picks the keypoints
                
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
        keypoints = IGEM_pick(gray_img, minArea, minCirc, minCon , minInert) ##picks the keypoints
                
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

def imgclass(inv_img):
    
    '''
    # uses a compressed grayscale image from cvt_color(RGB2GRAY)
    # returns the intensity histogram and related bins position w/ im_class
    # can optimize this function to a greater extent.
    
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    
    '''
    ##can edit to make a histogram from of the pixle image intensities of the image...
    
    hist, bins = np.histogram(inv_img.flatten(),256,[0,256])
    #bincenters = 0.5*(bins[1:]+bins[:-1])
    
     ##apending max histogram intensities into a list
    histx = np.argmax(hist)
        
    if histx < 110:
        
        im_class = 1
    
    elif 110 <= histx < 120:
        
        im_class = 2 
        
    elif 120 <= histx < 125:
        
        im_class = 3
        
    elif 125 <= histx < 130:
        
        im_class= 4
        
    elif 130 <= histx < 135:
        
        im_class= 5
    
    elif 135 <= histx < 140:
        
        im_class= 6
        
    elif 140 <= histx < 145:
        
        im_class= 7
        
    elif 145 <= histx < 150:
        
        im_class= 8
        
    elif 150 <= histx < 160:
        
        im_class= 9
    elif histx >= 160:
        
        im_class= 10
    
    return im_class, histx

def septest(p,image):
    
    '''
    let p be a range of integers ranging from [1, x], for the publication x
    is set to 31
    
    let image be a grayscale image produced after original image compression and 
    conversion to grayscale using OpenCv's function
    
    image = gray_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    
    
    '''
    
    detected_bin = np.zeros(len(p))
    detected_lap = np.zeros(len(p))
    detected_dog = np.zeros(len(p))
    
    #the background conditions of various image sets will varry - 
    #go back and plot 
    for i in range(len(p)): 
        
        #same scaling factor as used by SIFT on the simple scale
        output_bin, _ = BIN_filter(p[i], image)
        output_lap = IGEM_filter(p[i],image, 'no')
        output_dog = DOG_filter(p[i],image, 'no') 
        
        keypoints_bin = IGEM_pick(output_bin, 20, .83, .73, .73)
        keypoints_lap = IGEM_pick(output_lap, 20, .83, .73 , .73)
        keypoints_dog = IGEM_pick(output_dog, 20, .83, .73 , .73)
        
        if len(keypoints_lap) > 0:
            detected_lap[i] = len(keypoints_lap)
        else:   
           detected_lap[i] = 0
            
        
        if len(keypoints_dog) > 0:
            detected_dog[i] = len(keypoints_dog)
        else: 
            detected_dog[i] = 0
            
        if len(keypoints_bin)>0:   
            detected_bin[i] = len(keypoints_bin)
        else: 
            detected_bin[i] = 0

    
    
    return detected_bin, detected_lap, detected_dog 

#returns an array of the number of particles detected per filtering method...

## have yet to write a function for binary thresholding (incorporate that in efficiency_test.py)
## Need - image compress function
## then clarification of tests that include those testing for efficiency and separation power of algorithm...

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


def fitpcf(data):
    
    '''
    data1 = pd.read_csv('/home/joseph/Documents/PHY479/pcf-dr5-error.csv', header=None, skiprows=1)
    Function initially created to plot graphs from V30M and CD1 positve controls ()
    please add modifications and change to suit your needs.
    
    **Note: pcf-dr5-error.csv is a file outputted from keypoints2pcf()
    look to that function to see how that output is formatted. 
    
    Output : One graph, with fitted curve for V30M data vs CD1 Data  
    Equation fitted to probability distribution for Complete Spatial Randomness 
    
    '''
   
    data = pd.DataFrame(data)
    data = data.fillna(0)

    #determine guess filtering parameters
    pcfp1 = np.array([100.,1.,1.])
    pcfp2 = np.array([10.,1., 1.])
    
    x = data[2].values
    y = data[0].values
    dy = data[1].values

    x1 = data[5].values
    y1 = data[3].values
    dy1 = data[4].values

    popt1, pcov1 = opt.curve_fit(pcf , x, y,  p0 = pcfp1)
    popt2, pcov2 = opt.curve_fit(pcf , x1, y1,  p0 = pcfp2)

    popt1 = np.around(popt1, decimals=2)
    popt2 = np.around(popt2, decimals=2)

    #The probability of locating the N t h {\displaystyle N^{\mathrm {th} }} 
    #N^{{{\mathrm {th}}}} neighbor of any given point, at some radial distance r 
    #{\displaystyle r} r is:
    
    plt.figure()
    plt.title('Probability of Gold Particle Colocolization on TTR micrographs' )
    #CSR of CD1 Micgrgrap set 
    plt.plot(x,y,'xr') #keypoints of CD1 micrographs
    plt.plot(np.arange(0,110,1), pcf(np.arange(0,110,1), popt1[0], popt1[1], popt1[2]),
                       'r-', label='CD1 CSR, N = {} +/- {}, L = {} +/- {}'.format(popt1[0],
                                                 np.around(np.sqrt(pcov1[0,0]), decimals=3),
                                                 popt1[1], np.around(np.sqrt(pcov1[1,1]), decimals=3))) 
    plt.errorbar(x, y, yerr=dy, fmt='xr')
    plt.plot(x1,y1, 'og') ##keypoints of V30M micrographs
    plt.plot(np.arange(0,110,1), pcf(np.arange(0,110,1), popt2[0], popt2[1], popt2[2]),
                       'g-', label='V30M CSR, N = {} +/- {}, L = {} +/- {}'.format(popt2[0], 
                                                  np.around(np.sqrt(pcov2[0,0]), decimals=3),
                                                  popt2[1], np.around(np.sqrt(pcov2[1,1]), decimals=3))) 
    plt.errorbar(x1, y1, yerr=dy1, fmt='og')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Radius (r)')
    #Probability Nth point at distance r 
    plt.ylabel('P(r)')