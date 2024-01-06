import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats:
        an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels:
        an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats:
        an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type:
        the name of a kernel type. 'linear' or 'RBF'.

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    
    # Your code here. You should also change the return value.
    
    num_categories = categories.shape[0]
    num_test = test_image_feats.shape[0]
    
    scores = []
    if kernel_type == "linear":
        classify = svm.LinearSVC()
    if kernel_type == "RBF":
        classify = svm.SVC(kernel='rbf')         
    
    for i in range(num_categories):
        new_labels = 2*(train_labels == categories[i])-1
        classify.fit(train_image_feats, new_labels)    
        cat_decision = classify.decision_function(test_image_feats)
        scores.append(cat_decision)
    
    scores = np.array(scores)
    scores = np.swapaxes(scores, 0, 1).argmax(axis = 1)#scores.reshape(scores.shape[1], scores.shape[0]).argmax(axis = 1)

    result = np.array(range(num_test), dtype=object)
    for i in range(num_categories):
        result[scores==i] = categories[i]
        
    return result