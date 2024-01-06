import cv2
import numpy as np


def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    # Placeholder that you can delete -- random points
    #x = np.floor(np.random.rand(500) * np.float32(image.shape[1]))
    #y = np.floor(np.random.rand(500) * np.float32(image.shape[0]))
    
    a = 0.06
    threshold = 0.005
    stride = 2
    sigma = 0.2
    rows = image.shape[0]
    cols = image.shape[1]
    xs = []
    ys = []
   
    I_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    I_y = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)
    
    oneD_Gaussian = cv2.getGaussianKernel(5, sigma=0)
    twoD_Gaussian = oneD_Gaussian*oneD_Gaussian.T
    
    I_x = cv2.filter2D(I_x, -1, twoD_Gaussian)
    I_y = cv2.filter2D(I_y, -1, twoD_Gaussian)

    Ixx = I_x**2
    Ixy = I_y*I_x
    Iyy = I_y**2

    # find the sum squared difference (SSD)
    for y in range(0,rows-descriptor_window_image_width,stride):
        for x in range(0,cols-descriptor_window_image_width,stride):
            Sxx = np.sum(Ixx[y:y+descriptor_window_image_width+1, x:x+descriptor_window_image_width+1])
            Syy = np.sum(Iyy[y:y+descriptor_window_image_width+1, x:x+descriptor_window_image_width+1])
            Sxy = np.sum(Ixy[y:y+descriptor_window_image_width+1, x:x+descriptor_window_image_width+1])
            #Find determinant and trace, use to get corner response
            detH = (Sxx * Syy) - (Sxy**2)
            traceH = Sxx + Syy
            R = detH - a*(traceH**2)
            #If corner response is over threshold, it is a corner
            if R > threshold:
                xs.append(x + int(descriptor_window_image_width/2 -1))
                ys.append(y + int(descriptor_window_image_width/2 -1))
    return np.asarray(xs), np.asarray(ys)




    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000



def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)


    # Placeholder that you can delete. Empty features.
        # Convert to integers for indexing 


    # Define helper functions for readabilty and avoid copy-pasting
    def get_window(y, x):
        """
         Helper to get indices of the feature_width square
        """
        rows = (x - (descriptor_window_image_width/2 -1), x + descriptor_window_image_width/2)
        if rows[0] < 0:
            rows = (0, rows[1] - rows[0])
        if rows[1] >= image.shape[0]:
            rows = (rows[0]  + (image.shape[0] -1 - rows[1]), image.shape[0] - 1)
        cols = (y - (descriptor_window_image_width/2 -1), y + descriptor_window_image_width/2)
        if cols[0] < 0:
            cols = (0, cols[1] - cols[0])
        if cols[1] >= image.shape[1]:
            cols = (cols[0]  - (cols[1] + 1 - image.shape[1]), image.shape[1] - 1)
        return int(rows[0]), int(rows[1]) + 1, int(cols[0]), int(cols[1]) + 1

    def get_current_window(i, j, matrix):
        """
        Helper to get sub square of size feature_width/4 
        From the square matrix of size feature_width
        """
        return matrix[int(i*descriptor_window_image_width/4):
                    int((i+1)*descriptor_window_image_width/4),
                    int(j*descriptor_window_image_width/4):
                    int((j+1)*descriptor_window_image_width/4)]

    def rotate_by_dominant_angle(angles, grads):
        hist, bin_edges = np.histogram(angles, bins= 36, range=(0, 2*np.pi), weights=grads)
        angles -= bin_edges[np.argmax(hist)]
        angles[angles < 0] += 2 * np.pi
    
    # Initialize features tensor, with an easily indexable shape
    features = np.zeros((len(x), 4, 4, 8))
    # Get gradients and angles by filters (approximation)
    sigma = 0.8
    
    oneD_Gaussian = cv2.getGaussianKernel(5, sigma)
    twoD_Gaussian = oneD_Gaussian * oneD_Gaussian.T
    
    filtered_image = cv2.filter2D(image, -1, twoD_Gaussian)
    dx = cv2.Scharr(image, cv2.CV_32F, 1, 0) / 15.36
    dy = cv2.Scharr(image, cv2.CV_32F, 0, 1) / 15.36
    gradient = np.sqrt(np.square(dx) + np.square(dy))
    angles = np.arctan2(dy, dx)
    angles[angles < 0 ] += 2*np.pi

    for n, (x, y) in enumerate(zip(x, y)):
        # Feature square 
        i1, i2, j1, j2 = get_window(x, y)
        grad_window = gradient[i1:i2, j1:j2]
        angle_window = angles[i1:i2, j1:j2]
        # Loop over sub feature squares 
        for i in range(int(descriptor_window_image_width/4)):
            for j in range(int(descriptor_window_image_width/4)):
                # Enhancement: a Gaussian fall-off function window
                current_grad = get_current_window(i, j, grad_window).flatten()
                current_angle = get_current_window(i, j, angle_window).flatten()
                features[n, i, j] = np.histogram(current_angle, bins=8,
                range=(0, 2*np.pi), weights=current_grad)[0]
                
    features = features.reshape((len(x), -1,))
    dividend = np.linalg.norm(features, axis=1).reshape(-1, 1)
    # Rare cases where the gradients are all zeros in the window
    # Results in np.nan from division by zero.
    dividend[dividend == 0 ] = 1
    features = features / dividend
    thresh = 0.25
    features[ features >= thresh ] = thresh
    features  = features ** 0.8
    # features = features / features.sum(axis = 1).reshape(-1, 1)
    return features

def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    # Placeholder random matches and confidences.
    matches = []
    confidences = []
    
    # Loop over the number of features in the first image
    for i in range(features1.shape[0]):
        # Calculate the euclidean distance between feature vector i in 1st image and all other feature vectors
        # second image
        distances = np.sqrt(((features1[i,:]-features2)**2).sum(axis = 1))

        # sort the distances in ascending order, while retaining the index of that distance
        ind_sorted = np.argsort(distances)
        # If the ratio between the 2 smallest distances is less than 0.8
        # add the smallest distance to the best matches
        if (distances[ind_sorted[0]] < 0.9 * distances[ind_sorted[1]]):
        # append the index of im1_feature, and its corresponding best matching im2_feature's index
            matches.append([i, ind_sorted[0]])
            confidences.append(1.0  - distances[ind_sorted[0]]/distances[ind_sorted[1]])
          # How can I measure confidence?
    confidences = np.asarray(confidences)
    confidences[np.isnan(confidences)] = np.min(confidences[~np.isnan(confidences)])     

    return np.asarray(matches), confidences
    
    
    
    

