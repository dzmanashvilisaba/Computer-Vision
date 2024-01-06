import os

import numpy as np
import time
import cv2
from get_image_paths import get_image_paths
from feature_extraction import feature_extraction

data_path = '../data'

categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding',
              'Street', 'Highway', 'OpenCountry', 'Coast', 'Mountain',
              'Forest']

abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

# Number of training examples per category to use.
num_train_per_cat = 100


train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)

images = []
for path in train_image_paths:
        images.append(cv2.imread(path)[:, :, ::-1])


t = images[0]
gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
print(type(gray))

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

grid_size = (16,16)
#padding = (8,8)
#locations = ((10,20),)

features = hog.compute(img = gray, winStride = grid_size)
print(len(features), len(features)%36)

features = features.reshape(int(len(features)/36), 36) 
print(features.shape)

