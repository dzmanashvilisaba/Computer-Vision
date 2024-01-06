import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_*.npy' exists and
    contains an vocab size x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """

    vocab = np.load(f'vocab_{feature}.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    def num_subdivisions(n):
        return int((1/3) * (math.pow(4, n + 1) - 1))
    
    def subdivide(image, n):
        queue = []
        queue.append(image)
        
        total = int((1/3) * (math.pow(4, n) - 1))
        for i in range(total): 
            img = queue.pop(0)

            left, right = np.array_split(img, 2, axis = 1)
            left_subs = np.array_split(left, 2)
            right_subs = np.array_split(right, 2)
            

            queue.append(left_subs[0])
            queue.append(left_subs[1])
            queue.append(right_subs[0])
            queue.append(right_subs[1])

        return queue

    len_img = len(image_paths)
    dim = vocab_size * num_subdivisions(max_level)
    image_features = np.zeros((len_img, dim))
     

    for i in range(len_img):
        image = cv2.imread(image_paths[i])[:, :, ::-1]
        total_histograms = []
        #   Do the pyramid thing
        for j in range(max_level+1):
            subdivisions = subdivide(image,j)
           
            if j == 0: weight = math.pow(2, -max_level)
            else: weight = math.pow(2, -max_level + j - 1)
            
            for sub_image in subdivisions:
                features = feature_extraction(sub_image, feature)
                
                dist = pdist(vocab, features)
                dist = np.argmin(dist, axis = 0)
                
                histo, bins = np.histogram(dist, range(vocab_size+1))
                norm = np.linalg.norm(histo)
                total_histograms.append(weight * histo/norm)
           
        image_features[i, :] = np.concatenate(np.array(total_histograms))
        #print(image_features.shape)
    return image_features


