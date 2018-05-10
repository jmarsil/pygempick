"""
Created on Wed May  9 15:06:15 2018

@author: joseph
"""
import numpy as np
import cv2

def compress(orig_img):
    '''
    Takes a large image and compresses it 3.3 times in our case 
    images are outputed originally as large 9MB images...
    
    (That much reolution is unecessary when determining positioning of gold 
    particles. )
    '''
    
    r = 1018/orig_img.shape[1] ##correct aspect ratio of image to prevent distortion
    dim = (1018, int(orig_img.shape[0]*r))
    
    resized_img = cv2.resize(orig_img, dim, interpolation = cv2.INTER_AREA)
    
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    
    return gray_img


##define the filter of the scaled 3x3 laplacian kernel...(High Contract LPF)
def igem_filt(p,image, noise):
    
    '''
    New High Contrast Laplace Filter. Takes odd scaling parameter p > 5, regular
    compressed image, if noise == 'yes' will add median blur after filter applied.
    
    '''
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    kernel = np.array([[0,-1,0], [-1,p,-1], [0,-1,0]])
    
    output = cv2.filter2D(gray_img, -1, kernel)
    
    if noise == 'yes':
        
        output = cv2.medianBlur(output,9)
    
    return output

##define difference of Gaussian Filter as used w/ SIFT method...(Lowe,2004)
def dog_filt(tau,image, noise):
    
    '''
    Difference of Gaussian Filter. Takes odd scaling parameter tau, regular
    compressed image, if noise == 'yes' will add median blur after filter applied.
    ''' 
    
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
    
    '''
    Smart Binary Filtering. Uses the average gray pixel intensity value to determing 
    the starting threshold position. Takes odd scaling parameter p, 
    image = regular compressed image
    ''' 
    
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
    image = resulting binary image from filter, minArea = lowest area in pixels 
    of gold particle (20 px**2), lowest circularity of gold particle [.78 is square], 
    minConv = lowest convextivity parameter (is there a space between detected 
    particle - optimization with this parameter can help us differentiate between
    overlapping particles, minINER = minimum inertial ratio (filters particle
    based on their eliptical properties). 
    
    Detects immunogold particles on filtered binary image by optimizing picking
    across 4 main paramaters using OpenCv's simple blob detector. 
    
    Have to optimize for each set separately on a per class or per trial basis. 
    
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

def snapshots(folder, keypoints, gray_img, i):
    
    '''
    folder = folder location where snapshots will be saved, keypoints = output 
    from pick() , gray_img = compressed grayscale image, i = image number.
    
    Takes an compressed grayscale image and uses the detected keypoints as a marker
    to take a snapshot within a 100px radius of that gold particle's position. 
    
    '''
    
    if len(keypoints) > 0:

        #append x and y coordinates of keypoint center pixels
        ni = len(keypoints) #number of particles in image
        x = np.zeros(ni)
        y = np.zeros(ni)
        
        k = 0
        
        for keypoint in keypoints:
            ## save the x and y coordinates to a new array...
            
            x[k] = keypoint.pt[0]
            y[k] = keypoint.pt[1]
            
            k+=1
        
        j = 0 #index counter
        count = 0
        
        for points in x:
            
            xs = x[j]
            ys = y[j]
            k = int(xs) #sets the center point x value reference
            h = int(ys) #sets the center point y value reference
            
            ind = []
            
            for s in range(len(x)):
                
                rad = np.sqrt((k-int(x[s]))**2 + (h-int(y[s]))**2)
                
                
                if rad < 50: 
                    
                    ind.append(s)
            
            ## will delete the pair of points that had a radius less than the 
            ## snapshot that was taken...
            x = np.delete(x, ind)
            y = np.delete(y, ind)
            
            ##fix boundary conditions...
            
            refxmin = k - 50
            refxmax = k + 50
            refymin = h - 50
            refymax = h + 50
            
            if refxmin < 0:
                refxmin = 0
            
            if refymin < 0:
                refymin = 0
            
            if refxmax > np.shape(gray_img)[1]:
                refxmax = np.shape(gray_img)[1]
            
            if refymax > np.shape(gray_img)[0]:
                refymax = np.shape(gray_img)[0]
            
            
            ##take a snapshot of the aggregates w/in a 50px radius...
            a = gray_img[refymin:refymax,refxmin:refxmax]
            
            ##snapshot counter...
            count+=1
            
            #write snapshot to snapshot folder...
            cv2.imwrite('{}/im_{}_snap_{}.jpg'.format(folder, i,count), a)
            
            if j in ind:
                j = 0
                
            else:
                j+=1
            
            if len(x) == 0:
                
                break
    
    else:
        
        count = 0
    
    return count