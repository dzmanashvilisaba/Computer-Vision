#  Computer Vision Projects

# Image Filtering and Hybrid Images

To implement 2D convolution in my_filter2D(). filtering algorithm should:

1. Pad the input image with zeros.
2. Support grayscale and color images and arbitrary shaped odd-dimension filters (e.g., 7x9 filters but not 4x5 filters).
3. Return a filtered image which is the same resolution as the input image.

To create hybrid image sum a low-pass filtered version of a first image and a high-pass filtered version of a second image. We must tune a free parameter for each image pair to controls how much high frequency to remove from the first image and how much low frequency to leave in the second image.


## Results
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Image%20Filtering%20and%20Hybrid%20Images/result/high_frequencies.jpg)
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Image%20Filtering%20and%20Hybrid%20Images/result/low_frequencies.jpg)
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Image%20Filtering%20and%20Hybrid%20Images/result/hybrid_image_scales.jpg)




#  Scene Recognition with Bag of Words

We will perform scene recognition with the bag of words method. We will classify scenes into one of 15 categories by training and testing on the 15 scene database.

To Implement scene recognition schemes:

1. Build vocabulary by k-means clustering (feature_extraction.py).
2. Principle component analysis (PCA) for vocabulary (get_features_from_pca.py).
3. Bag of words representation of scenes (get_bags_of_words.py, get_spatial_pyramid_feats.py)
4. Multi-class SVM (svm_classify.py).



## Results
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Bag%20of%20Words/confusion_matrix.png)
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Bag%20of%20Words/Untitled.png)



# Stereo Matching

We will implement codes for image rectification and a basic stereo algorithm. 

To Implement rectification and stereo algorithms:

1. Interpolate image using bayer image interpolation.
2. Find fundamental matrix using the normalized eight point algorithm. 
3. Rectify stereo images by applying homography.
4. Make a cost volume using zero-mean NCC(normalized cross correlation) matching cost function for the two rectified images, then obtain disparity map from the cost volume after aggregate it with a box filter. (calculate_disparity_map)




## Results
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Stereo%20Matching/result/rectified_anaglyph.png)
![](https://github.com/dzmanashvilisaba/Computer-Vision/blob/main/Stereo%20Matching/result/disparity_map.png)

