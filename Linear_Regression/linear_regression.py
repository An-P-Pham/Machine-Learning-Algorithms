import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # 1: Fill in your code here                         #
    #####################################################
    mse = (X.dot(w) - y) ** 2 #square error
    return sum(mse)/len(y) #means it

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	2: Fill in your code here                         #
  #####################################################		
  test = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
  w = np.dot(test, y)
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # 4: Fill in your code here                         #
  #####################################################		
    test = np.dot(np.linalg.inv(np.dot(X.transpose(), X) + (lambd * np.identity(X.shape[1]))), X.transpose())
    w = np.dot(test, y)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # 5: Fill in your code here                         #
    #####################################################		
    tuners = [2**(-1*i) for i in range(15)] #list comprehension of tunner values
    tuners = tuners[::-1] #reverse the list
    firstIt = True
    bestlambda = None
    minMSE = None
    for lambd in tuners:
        model_ith = regularized_linear_regression(Xtrain, ytrain, lambd)
        mse_ith = mean_square_error(model_ith, Xval, yval)
        if firstIt == True:
            minMSE = mse_ith
            firstIt = False
            bestlambda = lambd
        else:
            if mse_ith <= minMSE:
                minMSE = mse_ith
                bestlambda = lambd

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    #  6: Fill in your code here                        #
    #####################################################		
    
    res = X
    for i in range(2,p+1):
        newX = X ** i
        for j in range(X.shape[1]):
            res = np.insert(res, res.shape[1], newX[:,j], axis=1)
    return res #np.array(newX)

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

