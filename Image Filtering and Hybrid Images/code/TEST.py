import cv2
import numpy as np
import time
import os
from my_filter2D import my_filter2D


def hw2_testcase():
    # This script has test cases to help you test your my_filter2D() function. You should verify here that your
    # output is reasonable before using your my_filter2D to construct a hybrid image in hw2.py. The outputs are all
    # saved and you can include them in your writeup. You can add calls to cv2.filter2D() if you want to check that
    # my_filter2D() is doing something similar.
    #
    # Revised by Dahyun Kang and originally written by James Hays.

    ## Setup
    test_image = cv2.imread('../data/cat.bmp', -1) / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )

    result_dir = '../result/test'
    os.makedirs(result_dir, exist_ok=True)

    cv2.imshow('test_image', test_image)
    cv2.waitKey(10)

    ##################################
    ## Identify filter
    # This filter should do nothing regardless of the padding method you use.
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    identity_image = my_filter2D(test_image, identity_filter)

    cv2.imshow('identity_image', identity_image)
    cv2.imwrite(os.path.join(result_dir, 'identity_image.jpg'), identity_image * 255)

    ##################################
    ## Small blur with a box filter
    # This filter should remove some high frequencies
    # blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # blur_filter = blur_filter / sum(sum(blur_filter))  # making the filter sum to 1

    # blur_image = my_filter2D(test_image, blur_filter)

    # cv2.imshow('blur_image', blur_image)
    # cv2.imwrite(os.path.join(result_dir, 'blur_image.jpg'), blur_image * 255)


    print('Press any key ...')
    cv2.waitKey(0)


if __name__ == '__main__':
    hw2_testcase()
