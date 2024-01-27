################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from utils import normalize_points
import numpy as np
import cv2
import itertools

from skimage.transform import FundamentalMatrixTransform

#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 7
DISPARITY_RANGE = 40
AGG_FILTER_SIZE = 5



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    ################################################################
    
    #print("Bayer Image is {}".format(bayer_img))
    
    h, w = bayer_img.shape 
    

    
    
    redChannel = np.zeros((h,w))
    blueChannel = np.zeros((h,w))
    greenChannel = np.zeros((h,w))
    
    ### Extracting channels
    redChannel[0:h:2, 0:w:2] = bayer_img[0:h:2, 0:w:2]
    blueChannel[1:h:2, 1:w:2] = bayer_img[1:h:2, 1:w:2]
    for i in range(2):
        greenChannel[i:h:2, 1-i:w:2] = bayer_img[i:h:2, 1-i:w:2] 
    
    #print("RED\n", redChannel)
    #print("GREEN\n", greenChannel)
    #print("BLUE\n", blueChannel)
    
    
    
    
    ### Define function of interpolation
    # def shifted(arr, horizontal, vertical):
        # h, w = arr.shape
        # newArray = arr
        
        # if horizontal < 0: 
            # horizontal = - horizontal
            # newArray = newArray[:, horizontal:w]
            # zeros = np.zeros((h, horizontal))
            # newArray = np.concatenate((newArray.T, zeros.T)).T
        # else: 
            # newArray = newArray[:, 0:w-horizontal]
            # zeros = np.zeros((h, horizontal))
            # newArray = np.concatenate((zeros.T, newArray.T)).T
        # if vertical < 0: 
            # vertical = -vertical
            # newArray = newArray[vertical:h, :]
            # zeros = np.zeros((vertical, w))
            # newArray = np.concatenate((newArray, zeros))
        # else: 
            # newArray = newArray[0:h-vertical, :]
            # zeros = np.zeros((vertical, w))
            # newArray = np.concatenate((zeros, newArray))
        # return newArray
        
    ## Helper shift operation for matrix #######################################################
    ## Example: shift with (+1,+1)
    ##     Pre                     Post
    ##     | A _ B _ ... |         | _ _ _ _ ... |
    ##     | _ _ _ _ ... |   -->   | _ A _ B ... |
    ##     | C _ D _ ... |   -->   | _ _ _ _ ... |
    ##     | _ _ _ _ ... |         | _ C _ D ... |
    
    def shifted(arr, horizontal, vertical):
        h, w = arr.shape
        pad_h, pad_w = np.absolute(horizontal), np.absolute(vertical)

        newArray = np.zeros((h + 2*pad_h, w + 2*pad_w))    
        newArray[pad_h + horizontal : h + pad_h + horizontal, pad_w + vertical:w + pad_w + vertical] += arr

        return newArray[pad_h:h+pad_h, pad_w:w+pad_w]

    ## Interpolation for Red channels #######################################################
    ##     Phase First:
    ##     | R _ R _ ... |         | R _ R _ ... |
    ##     | _ _ _ _ ... |   -->   | _ R _ R ... |
    ##     | R _ R _ ... |   -->   | R _ R _ ... |
    ##     | _ _ _ _ ... |         | _ R _ R ... |
    Red =  shifted(redChannel, -1, 1) + shifted(redChannel, 1, 1) + shifted(redChannel, -1, -1) + shifted(redChannel, 1, -1)
    
    redChannel[1:h-1:2, 1:w-1:2] = Red[1:h-1:2, 1:w-1:2]/4
    if h%2 == 0: redChannel[h-1, 1:w-1:2] = Red[h-1, 1:w-1:2]/2
    if w%2 == 0: redChannel[1:h-1:2, w-1] = Red[1:h-1:2, w-1]/2
    if h%2 == 0 and w%2 == 0: redChannel[h-1, w-1] = Red[h-1, w-1]
    ##     Phase Second:
    ##     | R _ R _ ... |         | R R R R ... |
    ##     | _ R _ R ... |   -->   | R R R R ... |
    ##     | R _ R _ ... |   -->   | R R R R ... |
    ##     | _ R _ R ... |         | R R R R ... |
    Red = shifted(redChannel, 1, 0) + shifted(redChannel, -1, 0) + shifted(redChannel, 0, -1) + shifted(redChannel, 0, 1)
    
    redChannel[0, 1:w-1:2] = Red[0, 1:w-1:2]/3
    if w%2 == 0: redChannel[0, w-1] = Red[0, w-1]/2
    redChannel[1:h-1:2,0] = Red[1:h-1:2,0]/3
    if h%2 == 0: redChannel[h-1, 0] = Red[h-1, 0]/2
    redChannel[1:h-1:2, 2:w-1:2] = Red[1:h-1:2, 2:w-1:2]/4
    redChannel[2:h-1:2, 1:w-1:2] = Red[2:h-1:2, 1:w-1:2]/4
    w_mod2, h_mod2 = w%2, h%2
    redChannel[h-1, h_mod2:w-1:2] = Red[h-1, h_mod2:w-1:2]/3
    if h_mod2 == 0: redChannel[h-1, 0] = Red[h-1, 0]/2
    redChannel[w_mod2:h-1:2, w-1] = Red[w_mod2:h-1:2, w-1]/3
    if w_mod2 == 0: redChannel[0, w-1] = Red[0, w-1]/2
    if (w_mod2 == 0 and h_mod2 == 1) or (w_mod2 == 1 and h_mod2 == 0): redChannel[h-1, w-1] = Red[h-1, w-1]/2
    
        
    ## Interpolation for Blue channels #####################################################
    ##     Phase First:
    ##     | _ _ _ _ ... |         | B _ B _ ... |
    ##     | _ B _ B ... |   -->   | _ B _ B ... |
    ##     | _ _ _ _ ... |   -->   | B _ B _ ... |
    ##     | _ B _ B ... |         | _ B _ B ... |
    Blue =  shifted(blueChannel, -1, 1) + shifted(blueChannel, 1, 1) + shifted(blueChannel, -1, -1) + shifted(blueChannel, 1, -1)
        

    
    blueChannel[2:h-1:2,2:w-1:2] = Blue[2:h-1:2,2:w-1:2]/4
    if h%2 == 1: blueChannel[h-1, 2:w-1:2] = Blue[h-1, 2:w-1:2]/2
    if w%2 == 1: blueChannel[2:h-1:2, w-1] = Blue[2:h-1:2, w-1]/2
    if h%2 == 1 and w%2 == 1: blueChannel[h-1, w-1] = Blue[h-1, w-1]
    blueChannel[0, 2:w-1:2] = Blue[0, 2:w-1:2]/2
    blueChannel[2:h-1:2, 0] = Blue[2:h-1:2, 0]/2
    if w%2 == 1: blueChannel[0,w-1] = Blue[0, w-1]
    if h%2 == 1: blueChannel[h-1,0] = Blue[h-1, 0]
    blueChannel[0,0] = Blue[0,0]
    ##     Phase Second:
    ##     | B _ B _ ... |         | B B B B ... |
    ##     | _ B _ B ... |   -->   | B B B B ... |
    ##     | B _ B _ ... |   -->   | B B B B ... |
    ##     | _ B _ B ... |         | B B B B ... |
    Blue = shifted(blueChannel, 1, 0) + shifted(blueChannel, -1, 0) + shifted(blueChannel, 0, -1) + shifted(blueChannel, 0, 1)
    
    blueChannel[0, 1:w-1:2] = Blue[0, 1:w-1:2]/3
    if w%2 == 0: blueChannel[0, w-1] = Blue[0, w-1]/2
    blueChannel[1:h-1:2,0] = Blue[1:h-1:2,0]/3
    if h%2 == 0: blueChannel[h-1, 0] = Blue[h-1, 0]/2
    blueChannel[1:h-1:2, 2:w-1:2] = Blue[1:h-1:2, 2:w-1:2]/4
    blueChannel[2:h-1:2, 1:w-1:2] = Blue[2:h-1:2, 1:w-1:2]/4
    w_mod2, h_mod2 = w%2, h%2
    blueChannel[h-1, h_mod2:w-1:2] = Blue[h-1, h_mod2:w-1:2]/3
    if h_mod2 == 0: blueChannel[h-1, 0] = Blue[h-1, 0]/2
    blueChannel[w_mod2:h-1:2, w-1] = Blue[w_mod2:h-1:2, w-1]/3
    if w_mod2 == 0: blueChannel[0, w-1] = Blue[0, w-1]/2
    if (w_mod2 == 0 and h_mod2 == 1) or (w_mod2 == 1 and h_mod2 == 0): blueChannel[h-1, w-1] = Blue[h-1, w-1]/2





    ## Interpolation for Green channels #####################################################
    ##     | _ G _ G ... |         | G G G G ... |
    ##     | G _ G _ ... |   -->   | G G G G ... |
    ##     | _ G _ G ... |   -->   | G G G G ... |
    ##     | G _ G _ ... |         | G G G G ... |
    Green = shifted(greenChannel, 1, 0) + shifted(greenChannel, -1, 0) + shifted(greenChannel, 0, -1) + shifted(greenChannel, 0, 1)
    
    greenChannel[0,0] = Green[0,0]/2
    greenChannel[0, 2:w-1:2] = Green[0, 2:w-1:2]/3
    if w%2 == 1: greenChannel[0, w-1] = Green[0, w-1]/2
    greenChannel[2:h-1:2,0] = Green[2:h-1:2,0]/3
    if h%2 == 1: greenChannel[h-1, 0] = Green[h-1, 0]/2
    greenChannel[1:h-1:2, 1:w-1:2] = Green[1:h-1:2, 1:w-1:2]/4
    greenChannel[2:h-1:2, 2:w-1:2] = Green[2:h-1:2, 2:w-1:2]/4
    w_mod2, h_mod2 = w%2, h%2
    greenChannel[h-1, (1-h_mod2):w-1:2] = Green[h-1, (1-h_mod2):w-1:2]/3
    if h_mod2 == 1: greenChannel[h-1, 0] = Green[h-1, 0]/2
    greenChannel[(1-w_mod2):h-1:2, w-1] = Green[(1-w_mod2):h-1:2, w-1]/3
    if w_mod2 == 1: greenChannel[0, w-1] = Green[0, w-1]/2
    if (w_mod2 == 0 and h_mod2 == 0) or (w_mod2 == 1 and h_mod2 == 1): greenChannel[h-1, w-1] = Green[h-1, w-1]/2    




    ### Converting channels for color channels
    redChannel = redChannel.astype(np.uint8)
    blueChannel = blueChannel.astype(np.uint8)
    greenChannel = greenChannel.astype(np.uint8)
    

    combinedChannels =  cv2.merge([redChannel, greenChannel, blueChannel])
    #cv2.imshow('img1_bayer', combinedChannels)
    #cv2.waitKey(0)
       
    ################################################################
    return combinedChannels



#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ################################################################
    rgb_img = None


    ################################################################
    return rgb_img


   
#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################

    ### Normalizing points with given utility functions
    #pts1, T1 = normalize_points(pts1, pts1.shape[0])
    #pts2, T2 = normalize_points(pts2, pts2.shape[0])
    
    ### Put vectors in matrix
    A = np.array([])
    for i in range(pts2.shape[0]):
        row = np.outer([pts1[i][0], pts1[i][1], 1], [pts2[i][0], pts2[i][1], 1]).reshape(9)
        A = np.append(A, [row])
    A = A.reshape(8,9)    
    
    ### Solve the fundamental matrix with SVD decomposition
    ATA = np.matmul(np.transpose(A),A)
    eigValues, eigVectors = np.linalg.eig(ATA)
    
    f = eigVectors[8]
    fundamental_matrix = f.reshape(3,3)
    
    ### Enforcing rank 2 of the fundamental matrix
    u, s, v = np.linalg.svd(fundamental_matrix)
    s[2] = 0

    fundamental_matrix = np.matmul(np.matmul(u,np.diag(s)),v)
    ### Un-normalizing the fundamental matrix
    #print("\n\nShapes are: \n \t T2:\t{},\n\tFundamental Matrix:\t{}\n\tT1:\t{}".format(T2.shape,fundamental_matrix.shape,T1.shape))
    
    #fundamental_matrix = np.matmul(np.matmul(np.transpose(T2), fundamental_matrix.resize(9)), T1)
    ################################################################
    
    #r = fundamental_matrix
    r = cv2.findFundamentalMat(pts1,pts2,2)[0]
    
    return r



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################
    # print(h1, h2)
    h = cv2.getPerspectiveTransform(np.array([[50,50], [150,50], [150,150], [50,150]], dtype="float32"), np.array([[1,1], [100,4], [132,88], [12, 150]], dtype="float32"))

    img1_rectified = cv2.warpPerspective(img1, h1, (img1.shape[1], img1.shape[0]))
    img2_rectified = cv2.warpPerspective(img2, h2, (img2.shape[1], img2.shape[0]))

#    img1_rectified, img2_rectified = None, None
    ################################################
    
    #cv2.stereoRectifyUncalibrated(
    
    
    return img1_rectified, img2_rectified




#=======================================================================================
def calculate_disparity_map(img1, img2):
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # You have to get disparity (depth) of img1 (left)
    # i.e., I1(u) = I2(u + d(u)),
    # where u is pixel positions (x,y) in each images and d is dispairty map.
    # Your code here
    ################################################
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    disparity_map = stereo.compute(img1_gray, img2_gray)
    
    #disparity_map = None
    


    ################################################################
    return disparity_map


#=======================================================================================
# Anything else:
