




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

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        w_shape = w.shape[0]

        y_convert = y.copy()
        y_convert[y_convert == 0] = -1
        for i in range(0,max_iterations):
            total_w = np.zeros((w_shape,), dtype=float)
            total_b = 0.0
            for j in range(0,N):
                temp = y_convert[j]*(np.dot(w.transpose(),X[j])+b)
                if temp<=0:
                    total_w = total_w+(y_convert[j]*X[j])
                    total_b = total_b+y_convert[j]
                
            w = w+((step_size/N)*total_w)
            b = b+((step_size/N)*total_b)

        
        

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        w_shape = w.shape[0]

        y_convert = y.copy()
        y_convert[y_convert == 0] = -1
        for i in range(0,max_iterations):
            total_w = np.zeros((w_shape,), dtype=float)
            total_b = 0.0
            for j in range(0,N):
                temp = y_convert[j]*(np.dot(w.transpose(),X[j])+b)
                P = sigmoid(-temp)
                total_w = total_w+(P*(y_convert[j]*X[j]))
                total_b = total_b+(P*y_convert[j])

            w = w+((step_size/N)*total_w)
            b = b+((step_size/N)*total_b)

        

    else:
        raise "Undefined loss function."

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
    value = 1.0/(1+np.exp(-z))
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
    pred_list = []
    for i in range(0,N):
        temp = np.dot(w.transpose(),X[i])+b
        if temp>0:
            pred_list.append(1)
        else:
            pred_list.append(0)

    preds = np.array(pred_list)
    
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

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################			
            x_n = X[n]
            y_n = y[n]
            wx_b = np.dot(x_n, np.transpose(w))+b
            max_z = np.max(wx_b)
            z_prep = wx_b-max_z
            upper = np.exp(z_prep)
            lower = np.sum(upper)
            prob = upper/lower
            
            prob[y_n] = prob[y_n]-1
            grad_w = np.zeros((C, D))
            grad_b = np.zeros(C)

            for k_prime in range(0,C):
                grad_w[k_prime] = prob[k_prime]*x_n
                grad_b[k_prime] = prob[k_prime]
            
            w = w-(step_size*grad_w)
            b = b-(step_size*grad_b)
        
        

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        one_hot = np.zeros((N,C))
        one_hot[np.arange(N),y]=1
        for it in range(max_iterations):
            wx_b = np.dot(X, np.transpose(w))+b
            max_z = np.max(wx_b)
            z_prep = wx_b-max_z
            upper = np.exp(z_prep)
            denom = np.sum(upper, axis = 1).reshape(N,1)
            prob = upper/denom
            grad_w = np.dot(np.transpose(prob-one_hot), X)
            grad_b = np.sum(prob-one_hot, axis=0)
            w = w-((step_size/N)*grad_w)
            b = b-((step_size/N)*grad_b)
        
        

    else:
        raise "Undefined algorithm."
    

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
    temp1 = np.matmul(X, w.transpose())+b
    preds = temp1.argmax(axis=1)
    
    assert preds.shape == (N,)
    return preds




        