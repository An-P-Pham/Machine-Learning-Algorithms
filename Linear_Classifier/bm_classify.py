import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    #we must include the bias term
    #axis = 0 is the row
    #axis = 1 is the column
    N, D = X.shape  # updated shape
    transLabels = np.ones(X.shape[0])
    for ind in range(len(y)):
        if y[ind] == 0:
            transLabels[ind] = -1
    X = np.insert(X, 0, 1, axis=1) #bias will be in the beginning
    w = np.insert(w, 0, b)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss                  # 
        ################################################
        for i in range(max_iterations):
            ithPred = np.dot(X,w)
            ithPred = transLabels * ithPred
            for j, arr in enumerate(ithPred): #go through each prediction & transform back
                if arr <= 0:
                    ithPred[j] = 1
                else:
                    ithPred[j] = 0
            ithPred = ithPred * transLabels
            learn = np.dot(np.transpose(ithPred), X)
            w += learn * (step_size/N)  #average GD
        
        

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for i in range(max_iterations):
            ithPred = np.dot(X, w) #gives up prediction labels
            ithPred = transLabels * ithPred
            ithPred = sigmoid(-1 * ithPred) * transLabels
            learn = np.dot(np.transpose(ithPred), X)
            w += learn * (step_size/N)
        

    else:
        raise "Undefined loss function."

    b = w[0] #assign the bias term
    X = np.delete(X, 0, axis=1)
    w = np.delete(w, 0) #remove the constant weight for bias before returning
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1/(1+np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    #include the bias
    X = np.insert(X, 0, 1, axis=1)  # bias will be in the beginning
    w = np.insert(w, 0, b)
    res = np.dot(X, w)
    preds = np.zeros(N) #populate the labels
    for i, pred in enumerate(res):
        if pred > 0:
            preds[i] = 1 #correct the labels

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    
    # we must include the bias term
    # axis = 0 is the row
    # axis = 1 is the column
    X = np.insert(X, 0, 1, axis=1)  # bias will be in the beginning
    w = np.insert(w, 0, b, axis=1) #add each class's respective bias

    one_hot = None

    if one_hot is None:
        if gd_type == "sgd":
            one_hot = np.zeros((1, C))
        elif gd_type == "gd":
            one_hot = np.zeros((N, C))
        else:
            raise "Undefined algorithm."

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################			
        for i in range(max_iterations):
            randIdx = np.random.randint(N)
            xith = X[randIdx]
            ithPred = np.dot(xith, np.transpose(w)) #produces Cx1 at the end
            ithPred -= np.max(ithPred)
            ithPred = np.exp(ithPred)
            ithPred = ithPred/np.sum(ithPred)
            ithPred[y[randIdx]] -= 1
            ithPred = np.reshape(ithPred, (w.shape[0], 1))
            xith = np.reshape(xith, (1, X.shape[1]))
            learn = np.dot(ithPred, xith)
            w += learn * (-1*step_size)
        

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        for i, arr in enumerate(y):
            one_hot[i][arr] = 1
        for i in range(max_iterations):
            ithPred = np.dot(X, np.transpose(w))
            ithPred -= np.amax(ithPred)
            ithPred = np.exp(ithPred)
            ithPred = ithPred/np.sum(ithPred, axis=1, keepdims=True)
            softmax = ithPred - one_hot
            learn = np.dot(np.transpose(softmax), X)
            w += learn * (-1* step_size / N)

    else:
        raise "Undefined algorithm."
    
    b = w[:, 0]
    X = np.delete(X, 0, axis=1)
    w = np.delete(w, 0, axis=1)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    X = np.insert(X, 0, 1, axis=1)  # bias will be in the beginning
    w = np.insert(w, 0, b, axis=1)  # add each class's respective bias
    calc = np.dot(X, np.transpose(w))
    preds = []
    for i in range(N):
        preds.append(np.argmax(calc[i]))
    preds = np.array(preds)
    assert preds.shape == (N,)
    X = np.delete(X, 0, axis=1)
    w = np.delete(w, 0, axis=1)
    return preds




        