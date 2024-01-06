import cv2
import numpy as np
import time
import os
# from my_filter2D import my_filter2D


def test():
    test_image = cv2.imread('../data/cat.bmp', -1) / 255.0
    test_image = cv2.resize(test_image, dsize=None, fx=0.7, fy=0.7, )

    result_dir = '../result/test'
    os.makedirs(result_dir, exist_ok=True)

    # cv2.imshow('test_image', test_image)
    cv2.waitKey(0)
    ##################################
    ## Identify filter
    # This filter should do nothing regardless of the padding method you use.
    
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    identity_image = my_filter2D(test_image, identity_filter)
    
    blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    blur_image = my_filter2D(test_image, blur_filter)

    cv2.imshow('identity_image', identity_image)
    cv2.imwrite(os.path.join(result_dir, 'identity_image.jpg'), identity_image * 255)
    cv2.waitKey(0)


def my_filter2D(image, kernel):
    # This function computes convolution given an image and kernel.
    # While "correlation" and "convolution" are both called filtering, here is a difference;
    # 2-D correlation is related to 2-D convolution by a 180 degree rotation of the filter matrix.
    #
    # Your function should meet the requirements laid out on the project webpage.
    #
    # Boundary handling can be tricky as the filter can't be centered on pixels at the image boundary without parts
    # of the filter being out of bounds. If we look at BorderTypes enumeration defined in cv2, we see that there are
    # several options to deal with boundaries such as cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, etc.:
    # https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    #
    # Your my_filter2D() computes convolution with the following behaviors:
    # - to pad the input image with zeros,
    # - and return a filtered image which matches the input image resolution.
    # - A better approach is to mirror or reflect the image content in the padding (borderType=cv2.BORDER_REFLECT_101).
    #
    # You may refer cv2.filter2D() as an exemplar behavior except that it computes "correlation" instead.
    # https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # correlated = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)   # for extra credit
    # Your final implementation should not contain cv2.filter2D().
    # Keep in mind that my_filter2D() is supposed to compute "convolution", not "correlation".
    #
    # Feel free to add your own parameters to this function to get extra credits written in the webpage:
    # - pad with reflected image content
    # - FFT-based convolution

    ################
    # h, w, c = image.shape
    # k = kernel.shape[0]
    
    # image = image * 255
    
    # print(image)
    # res = np.zeros((h,w,c))

    
    # for i in range(1,h-2*k-1):
        # for j in range(1,w-2*k-1):
            # for t in range(3):
                # res[i,j,t] = np.sum(image[i-1:i+2,j-1:j+2,t]*kernel)
    
    # res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
    # return res
    
    output = image.copy()
    h, w, c = image.shape
    ker_h, ker_w = kernel.shape
    pad_h, pad_w = int((ker_h - 1)/2), int((ker_w - 1)/2)
    
    pad_mat = np.zeros((h + 2*pad_h,w + 2*pad_w,3))
    pad_mat[pad_h: h + pad_h, pad_w: w + pad_w] = image

    for d in range(len(image[0][0])):
        for i in range(len(image)):
            for j in range(len(image[0])):
                output[i][j][d] = sum(sum(np.multiply(kernel, pad_mat[i:i + ker_h, j:j + ker_w, d])))
                 
    return output
    
    ################

test()
