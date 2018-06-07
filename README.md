# pyGemPick: Open Source Gold Particle Picker for Immunogold Diagnostics

### This is the official installation guide for the PyGemPick module 

PyGemPick is the cummulation of [Joseph Marsilla's](https://github.com/jmarsil) research project 
under Dr. Avi Chakrabartty. This module contains functions that enable 
filtering, detection, and modeling of immunogold particles on TEM micrographs. 

The main project goal was to greate an open source batch gold particle picking
module built in python that could detect gold particles regardless of the amount
of counterstaining present in the IGEM (Immunogold Electron Microscopy) micrograph. 

#### pyGemPick has three main dependencies that are needed before usage

	1. [OpenCV (cv2)](https://opencv.org/) 
	2. [Pandas (pd)](https://pandas.pydata.org/)
	3. [Numpy  (np)](http://www.numpy.org/) 

I would suggest installing a new anaconda environment using [anaconda](https://conda.io/docs/user-guide/getting-started.html)
terminal into which you can import all the required modules for your project. Having trouble installing OpenCv, use the solution 
outlined here: [(install using conda)](https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda).
Pandas and Numpy can also be installed through any terminal using _**pip install pandas, numpy**_

The project will be updated in the upcoming weeks with tutorials on how 
to use the functions given within pygempick. This module was built to
help researchers diagnose pateints with rare protein misfolding diseases 
like ATTR, AD, FTD and ALS using novel Immunogold diagnostic techniques. 

_**NEW:**_ This update contains supplementary 11 supplementary 
documents that will help you use the module. We cover image compression, 
image picking with singular and duplicate filtering, statistical analysis,
separation & efficiency tests to test the algorithm's useability. 

**Sample Image Data will be provided and shall be located in the DATA folder**

## Installation 
	
	pip install pygempick

    > import pygempick.core as py
    > import pygempick.modeling as mod
    > import pygempick.spatialstats as spa

#### Note numpy, pandas and opencv modules dependencies are needed prior installation. 

**NEW:** This update contains supplementary 11 supplementary 
documents that will help you use the module. We cover image compression, 
image picking with singular and duplicate filtering, statistical analysis,
separation & efficiency tests to test the algorithm's fairness. 

For more information visit the github!


## Functions:

* __*py.compress(orig-img)*__
    
    * a function that takes an original large scale electron 
    micrograph image and compresses it such that 1px = aproximately one nanometer. 
    the exact pixle dimentions for a 3.1x compression are given below.

* __*py.back_eq(image)*__
    
    * background equalization provided by solution presented 
    [here](https://stackoverflow.com/questions/39231534/get-darker-lines-of-an-image-using-opencv)

* __*py.hclap_filt(p,image, noise)*__
    
    * New High Contrast Laplace Filter. 
    * Applies a [HCLAP Kernel](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) 
    * Takes odd scaling parameter p > 5 with a regular compressed image
    * if noise == 'yes' will add median blur after filter applied.

* __*py.hlog_filt(p, image, noise)*__
    
    * New High-Contrast Laplace of Gaussian Filter. 
    * Applies a [HCLOG Kernel](http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html)  
    to each image to produce a single binary image as an output. 
    * Takes odd and even scaling # parameters 18+ 
    * input image is regular py.compress image output, 
    * if noise == 'yes' will add median blur after filter applied.
    
* __*py.dog_filt(p, image)*__
    
    * [Difference of Gaussian Filter](http://www.tjscientific.com/2017/01/31/using-python-and-opencv-to-create-a-difference-of-gaussian-filter/). Input is an odd number p to determine size of DOG kernel,
    * input is an py.compress output image, 
    * if noise == 'yes' will add median blur after filter applied.

* __*py.bin_filt(p, image)*__
    
    * Smart Binary Filtering. Uses the average gray pixel intensity values to determing 
    the starting [threshold position](https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html).
    * Takes odd scaling parameter p, input image is a py.compress output image
    
    _**Note:**_ TEM migrograph filtering using simple binary thresholding was first
    completed in 2003 with one of the first gold particle picking algorithms [GoldFinder](https://www.sciencedirect.com/science/article/pii/S104784770200624X).

* _**New: key_filt(keypoints1, keypoints2)**_
    
   * Allows you to scandetected keypoints and eliminate duplicates! Allows you 
    to detect partciles with more than one filter. Returns updated keypoints
    1 with the removed keypoints and number of duplicate(s) detected.

* _**py.pick(image, minAREA, minCIRC, minCONV, minINER, minTHRESH)**_
    
    * Input image is a binary image from one of the above filters, next have to set
    the parameters to optimize [OpenCv's Simple Blob Detector](https://www.learnopencv.com/blob-detection-using-opencv-python-c/)
    * Detects immunogold particles on filtered binary image by optimizing picking
      across 4 main paramaters using OpenCv's simple blob detector.
    * Have to optimize picking for each set separately on a per class or per trial basis. 
    
    ##### Gold Particle Picking Parameters
    
        1. minArea = lowest area in pixels of a detected gold particle (20 px**2)
        2. minCirc = lowest circularity value of a detected gold particle [.78 is square]
        3. minConv = lowest convextivity parameter which is  Convexity is defined as the (Area of the gold particle / Area of it’s convex hull)
        4. minINER = minimum inertial ratio (filters gold particles based on  eliptical properties, 1 is a complete circle)

* _**py.snapshots(folder, keypoints, gray_img, i)**_
    
    * folder = folder location where snapshots will be saved, keypoints = the detected
    keypoints from py.pick function , gray_img = compressed grayscale image, i = image number.
    
    * Takes an compressed grayscale image and uses the detected keypoints as a marker
    to take a snapshot of within a 100px radius of that gold particle's position.
    *Researchers use this to analyze the morphological properties of protein aggregates*
     
    
* _**mod.draw(n, test_number, noise, images)**_
    
   function to draws test micrograph sets that will be used in subsequent 
   efficiency or separation tests. 
    
    1. Test number 1 is draw only circles, 2 is draw both circles and ellipses. 
    2. Noise if == 'yes' then, randomly distibuted gaussian noise will be drawn 
        according to mu1, sig1. 
    3. images are the number of images in the set - used with n which is number of 
       particles detected in the actual set to calulate the particle density of model 
       set.

* _**mod.imgclass(inv_img)**_
    
    * Uses a compressed grayscale image from [cv2.cvt_color(RGB2GRAY)](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html)
    and returns the intensity histogram and related bins position w/ im_class. 

* _**mod.septest(p,image)**_
    
    * Let p be a range of integers ranging from [1, x], let image be a grayscale 
    image produced after original image compression and conversion to grayscale 
    using OpenCv's function cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY).
    
    * Completes separation test for single filter comparrison.

* _**New mod.septest2(p, image, hlogkey)**_
    
    * let p be a range of integers ranging from [1, x] , let image be a grayscale
    image produced after original image compression and conversion to grayscale
    using OpenCv's function  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY).
    
    * hlogkey = the keypoints of detected image fitered with HLOG filter - this ensures
    faster particle detection since we aren't running the same filtering step more 
    than once! 
    
    * Completes separation test for _**dual high-contrase filter comparrison**_.

* _**mod.fitpcf(data)**_
    
    * Data is the input from a csv created by sta.bin2csv
    * file is in format of pcf-dr#-error.csv'. 
    * Function initially created to plot graphs for image set 
    with varrying concentrations of AB aggregates in solution
    
    **Output:** built to produce one graph, with fitted curve for positive control(s).  
    Equation fitted to probability distribution for Complete Spatial Randomness of 
    the distribution of IGEM particles across EM micrographs.

* __*spa.gamma(a,b,r)*__

    * a = width of image in pixels
    * b = height of the image in pixels
    * r is the diatance of the donut from which correlation was calculated. 
    
    Function taken from work by [Philemonenko et al 2000](http://nucleus.img.cas.cz/pdf_publications/PHI_Statistical%20evaluation%20of%20Colocalization%20Patterns_01.pdf)
    that was used as a window covariogram to correct [Ripley's K function](http://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:ripley_s_k_and_pair_correlation_function) for boundary conditions.

* _**spa.pcf(r, N, p0, p1)**_
    
    * r is the radius of the donut taken with bin width dr. 
    * N is the degree PCF (Pair Correlation Function) is the probability distribution of a CSR 
      related process that we will used to fit our normalized version of 
    
    * This is a python based solution to [Philmoneko's Statistical Evaluation of Colocalization Patterns in Immunogold Labeling Experiments](http://nucleus.img.cas.cz/pdf_publications/PHI_Statistical%20evaluation%20of%20Colocalization%20Patterns_01.pdf). 
    The PCF distribution for calculating the colocolization of immunogold particles
     on transmission electorn microgrpahs is represented here.

* _**spa.record_kp(i, keypoints, data)**_

    * i is the image number counter
    * keypoints is the list of keypoints of Gold particles detected by py.pick
    * data is an empty pandas dataframe. 
    
    This function recods the x,y positions of the keypoints detected in each image. 
    Run in for loop to add results for each image to dataframe which can be then exported
    into a csv for easy access. (completed in spa.bin2csv )
    
* _**spa.bin2csv(images)**_

   * function takes a list of filelocations from glob.glob (asks for the
     filtering parameters) then it outputs a csv of the x and y coordinates of 
     keypoints for every image in images. (For example, row 1 contains the x 
     coordinate of the keypoints in image 1 and row 2 contains the y coordinates in image 1 ect...)

* _**spa.bin2df(images)**_
    
    * images is a set of images from folder using glob.glob() function,
    
    * Output records the keypoint positions found in each image and outputs a pandas
      df with detected keypoint centers in (x,y) pixel coordinates. 

* _**spa.csv2pcf(data, dr)**_

    * takes the filename `data` from a csv produced by bin2csv() and outputs 
      non-normalized scale invarient k (cross-corelation) and pcf (pair-correlation) 
      statisticaldata from the spatial distribution of the paticles on each micrograph.
      (determines wheter the nul-hypothesis of CSR [Complete Spatial Randomness](https://en.wikipedia.org/wiki/Complete_spatial_randomness) is 
      upheld or voided...). Analyzed by bin2csv. Example output provided in docs.
    
    * dr is the donut width as defined by philmonenko et al, 2000

* _**spa.keypoints2pcf(data_set, dr)**_
    
    * Input folder with CSV files of keypoints for different tests
      Need to know Image number and average particles detected in each set
      (**example**: data_set = glob.glob('/home/joseph/Documents/PHY479/Data/anto/*.csv'))
      
    * dr is the donut width as defined by [philmonenko et al, 2000 article](http://nucleus.img.cas.cz/pdf_publications/PHI_Statistical%20evaluation%20of%20Colocalization%20Patterns_01.pdf) 
      on immunogold particle colocolization and spatial statistcs. 
    
    * **output:** pcf-dr{}-error.csv - columns dr (sampling radius), pcf 
    ([pair correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)),
     dpcf (propogated uncertainty in pcf)
