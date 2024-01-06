import cv2
import numpy as np


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a number of grid points x feature_size matrix.
    """
    
    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        # Your code here. You should also change the return value.
        #   Convert image into gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #   Create HOG descriptor from open cv
        hog = cv2.HOGDescriptor(  _winSize = win_size, \
                            _blockSize = block_size, \
                            _blockStride = block_stride, \
                            _cellSize = cell_size, \
                            _nbins = nbins,
                            _derivAperture = deriv_aperture, \
                            _winSigma = win_sigma, \
                            _histogramNormType = histogram_norm_type, \
                            _L2HysThreshold = l2_hys_threshold, \
                            _gammaCorrection = gamma_correction, \
                            _nlevels = nlevels)
                            
        #grid_size = (16,16)
        #padding = (16, 16)
        #locations = ((10,20),)
        #   Compute the features
        features = hog.compute(gray)
        #   Resize the list of features to fit the required shape
        features = features.reshape(int(len(features)/36), 36) 

    
        return features
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 36.

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.
        #   Create SIFT from open cv
        sift = cv2.SIFT_create()
        #   Given size of grid from which we extract features
        grid_size = 20
        #   Convert into gray-scale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #   Set of keypoints of grid from which we exctract features
        kp = [cv2.KeyPoint(x, y, grid_size) for y in range(0, gray.shape[0], grid_size) \
                                    for x in range(0, gray.shape[1], grid_size)]
        #   Extracted features 
        keypoints, descriptors = sift.compute(gray, kp)
        
        return descriptors
        # `.shape[0]` do not have to be (and may not) 1500,
        # but `.shape[1]` should be 128.



