PyGemPick is the cummulation of Joseph Marsilla's research project 
under Dr. Avi Chakrabartty. This module contains functions that enable 
filtering, detection, and modeling of immunogold particles on TEM micrographs. 

The main project goal was to greate an open source batch gold particle picking
module built in python that could detect gold particles regardless of the amount
of counterstaining present in the IGEM (Immunogold Electron Microscopy) micrograph. 

This module has three main dependencies that are needed before usage: 

	1. OpenCV (cv2) for image processing. 
	2. pandas (pd) for Data analysis 
	3. numpy  (np) 


The project will be updated in the upcoming weeks with tutorials on how 
to use the functions given within pygempick. This algoritm was built to
help researchers diagnose pateints with rare protein misfolding diseases 
like ATTR, AD, FTD and ALS using novel Immunogold diagnostic techniques. 

--------

Index:

compress(orig-img) is a function that takes an original large scale electron 
micrograph image and compresses it such that 1px = aproximately one nanometer. 
the exact pixle dimentions for a 3.1x compression are given below - All 
micrographs in each set were taken at random with a 80Kv Joel 1200 microscope 
at 150,000x magnification.

IGEM_filter(p,image, noise) is a function that takes a scaling factor value
and applies the HCLAP filter to the compressed grayscale image produced by 
compress(orig-img). Output is the filtered binary image. If image was drawn by
our IGEM_draw function, add 'yes' to the noise section. This applies median 
filter to neutralize all the speckled noise that the filter doesnt take care of

DOG_filter(tau,image, noise) is a function that takes a scaling factor tau 
and outputs an image from the difference of gaussian method 
ie output = I*DOG - I*DOG*sqrt(2)*tau

BIN_filter(p, image) is a function that takes a regular grayscale image and 
runs binary thresholding from the difference of the average gray pixel intensity
 of the micrograph and 60 + 1.5*p , which is the scaling parameter that will 
 allow us to modulate the sepatation power of the binary filtering method

IGEM_pick(image, minAREA, minCIRC, minCONV, minINER) is the function that takes
a filtered binary image from BIN_filter(p, image), DOG_filter(tau,image, noise),
or IGEM_filter(p,image, noise) and four minfiltering parameters used to 
detect and filter out keypoints of interest. Values will varry depending on 
counter staining conditions that are present in a micrograph set. 

def IGEM_draw(n, test_number, noise, images, mu1, sig1) is a function to draw 
test micrograph sets that will be used in subsequent efficiency tests. 
    
    1. Test number 1 is draw only circles, 2 is draw both circles and ellipses. 
    2. Noise if == 'yes' then, randomly distibuted gaussian noise will be drawn 
        according to mu1, sig1. 
    3. images are the number of images in the set - used with n which is number of 
    particles detected in the actual set to calulate the particle density of model 
    set.

Gamma(r) is the window covariogram to solve the boundary problem experienced when
finding the original Ripley's K function... Takes radial components as well as the
width (a) and height (b) in pixels of the image. 

bin2csv(images) - function takes a list of filelocations from glob.glob (asks for the
filtering parameters),then it outputs a csv of the x and y coordinates of 
keypoints for every image in images. For example row 1 contains the x coordinate 
of the keypoints in image 1 and row 2 contains the y coordinates in image 2. 

bin2df(images) takes the images and instead of outputing ans saving a csv, it returns
a pandas dataframe. 

csv2pcf(data, dr) - takes the filename data from a csv produced by bin2csv() and outputs 
non-normalized scale invarient k (cross-corelation) and pcf (pair-correlation) 
statisticaldata from the spatial distribution of the paticles on each micrograph.
(determines wheter the nul-hypothesis of CSR [Complete Spatial Randomness] is 
upheld or voided...)
    
imgclass(inv_img) -  uses a compressed grayscale image from cvt_color(RGB2GRAY)
returns the im_class and the mean of the pixel intensity histogram of the grayscale
image being analyzed...

septest(p,image) - test the separation power of particles detected while changing the 
scaling factor of set filtering method, will return the particles picked and detected
for each scaling factor input...if if p = range(1,31,2) it will return two arrays
for particles detected at each filter scale. First position of picked lap outlines
the number of particles detected when that the first was used on the laplacian filter.

    1 = LAP (Modified Laplacian Filter), 
    2 = DOG (Normal Difference of Gaussian Filter) 

    Note: Binary filter is more of a morphological filtering technique outlined first in 
    2003 with the algorithm goldfinder (that approach doesn't use scales but intensity 
                                        thresholding)

keypoints2pcf(data_set, dr) is a function that takes the output from csv2pcf , propogates
the error and normalize the resulting pcf data which will be plotted by fitpcf()

pcf(r, N, p0, p1) is the probability distribution of a CSR related process 
that we will used to fit our normalized version of Philmoneko's PCF diostributtion
for calculating colocolization of immunogold particles on microgrpahs
 
fitpcf(data) is a function that takes the output of keypoints2pcf (CSV file)
and plots resulting normalized PCF for V30M and CD1 positive controls. **needs
modification if looking to test distributions for more than one set, but note 
resolution of calculation and plot will increase with number of images and 
detected keypoints. That's why out positive control sets were used because they
have an average particle density of 10 - 21 particles per image. Therefore 
the distribution of the colocolization can be easily calculated. 
