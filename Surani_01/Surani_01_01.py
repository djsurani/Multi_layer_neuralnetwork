# Surani, Dhruv
# 1002_039_855
# 2023_02_27
# Assignment_01_01

import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Arguments:
    x -- input value or array.

    Returns:
    output -- sigmoid activation of x.
    """

    output = 1 / (1 + np.exp(-x))  # Calculate sigmoid
    return output  # Return output


def initializes_weights(layer_sizes, seed):
    """
    Initializes the weights of a neural network.

    Arguments:
    layer_sizes -- list of integers representing the number of nodes in each layer.
    seed -- integer seed value for the random number generator.

    Returns:
    weights -- list of weight matrices, one for each layer.
    """

    weights = []
    for i in range(1, len(layer_sizes)):
        np.random.seed(seed)  # Set seed for random number generator
        weight = np.random.randn(layer_sizes[i], layer_sizes[i - 1] + 1)  # Generate random weights for layer i
        weights.append(weight)  # Add weight matrix to weights list
        
    return weights  # Return weights list

#forward pass 
def fwd_pass(x, weights):
    """
    Computes the forward pass of a neural network.

    Arguments:
    x -- input data as a numpy array.
    weights -- list of weight matrices, one for each layer.

    Returns:
    a -- final output of the neural network.
    layer_inps -- list of inputs to each layer.
    """

    layer_inps = []  # Initialize empty list for inputs to each layer
    a = x  # Set current layer input to x
    i = 0
    while i < len(weights):
        weight = weights[i]  # Get weight matrix for current layer   
        layer_inps.append(a)  # Add current layer input to list of inputs
        bias_row = np.ones((1, a.shape[1]))  # Create a row of ones to represent bias term
        a_with_bias = np.vstack([bias_row, a])  # Add bias term to input array
        z = np.dot(weight, a_with_bias) # Compute linear transformation
        a = sigmoid(z)  # Apply sigmoid activation function to get output of current layer
        i += 1
    return a, layer_inps  # Return final output and list of inputs to each layer

# find mean square error
def cal_mean_squr_err(y_predicted, y_t):
    """
    Computes the mean squared error between predicted and true outputs.

    Arguments:
    y_predicted -- predicted outputs as a numpy array.
    y_true -- true outputs as a numpy array.

    Returns:
    mse -- mean squared error between predicted and true outputs.
    """
    squared_error = (y_predicted - y_t) ** 2
    mse = np.mean(squared_error)  # Calculate mean squared error
    return mse  # Return the mean squared error


# forward pass with loss
def frwd_pass_loss(x, weights, y_t):
    """
    Computes the forward pass of the neural network and the loss function.

    Arguments:
    x -- input data as a numpy array.
    weights -- list of weight matrices for each layer.
    y_true -- true outputs as a numpy array.

    Returns:
    loss -- mean squared error between predicted and true outputs.
    """

    y_predicted, _ = fwd_pass(x, weights)  # Compute forward pass of the neural network
    loss = cal_mean_squr_err(y_predicted, y_t)  # Compute mean squared error between predicted and true outputs
    return loss  # Return the loss

#backward pass
def bcwd_pass(weights, layer_inps, y_t, h, alpha):
    # Find gradients using centered difference approximation
    pd = [] #partial_derivatives
    # Iterate over the different layers
    i = 0 
    while i < len(weights):
        curr_weight = np.copy(weights[i]) # Weight matrix for current layer
        downstream_weights = weights[i + 1:] # Weight matrices of downstream layers
        layer_input = layer_inps[i] # Input matrix fed to this
        pd_w = np.zeros_like(curr_weight) # Placeholder to store partial derivative for current 
        i += 1
        # Iterate over all the node elements in a weight and add perturbations in them of size h 
        for a in range(curr_weight.shape[0]):
            # loop throught each column of the current weight matrix
            for b in range(curr_weight.shape[1]):
                # create a copy of the cureent weight matrix
                weight_plus_h = np.copy(curr_weight)
                #perturb the current weight by adding h to the current element
                weight_plus_h[a, b] += h
                # Create a copy of the current weight matrix
                weight_minus_h = np.copy(curr_weight)
                # Perturb the current weight by subtracting h from the current element
                weight_minus_h[a, b] -= h
                # Create the input weights with the perturbed weights for both positive and negative perturbations
                inp_wei_wplush = [weight_plus_h] + downstream_weights
                inp_wei_wminus_h = [weight_minus_h] + downstream_weights
                 # Calculate the partial derivative for the current element using the centered difference approximation
                partial_der = (
                    frwd_pass_loss(layer_input, inp_wei_wplush, y_t)
                     - frwd_pass_loss(layer_input, inp_wei_wminus_h, y_t)
                ) / (2 * h)
                # Store the partial derivative for the current element in the partial derivative matrix
                pd_w[a, b] = partial_der
                
        pd.append(pd_w)
        
    ## Now, update weight matrices
    updated_weights = [] # create an empty list to store the updated weights
    # iterate over the weight matrices and their corresponding partial derivatives
    for weight, partial_derivative in zip(weights, pd):
         # compute the updated weight using gradient descent
        updated_weight = weight - alpha * partial_derivative
         # add the updated weight to the list
        updated_weights.append(updated_weight)
        
    return updated_weights # return the list of updated weights

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    inp_dim, num_samples = X_train.shape
    layer_num_nodes = layers.copy()
    layer_num_nodes.insert(0, inp_dim)
    
    # Initialize weights
    weights = initializes_weights(layer_num_nodes, seed)
    
    # Placeholder for recording errors
    errors = []
    epoch = 0 # initialize epoch counter 
    while epoch < epochs: # continue looping until epoch reaches the desired number of epochs
        y_predicted_train, layer_inputs = fwd_pass(X_train, weights) # calculate forward pass for training set
        t_error = cal_mean_squr_err(y_predicted_train, Y_train) #calculate training error
        weights = bcwd_pass(weights, layer_inputs, Y_train, h, alpha) #update weights for backward pass 
        
        y_predicted_test, _ = fwd_pass(X_test, weights) # test set's forward pass
        test_error = cal_mean_squr_err(y_predicted_test, Y_test) # cal test error
        errors.append(test_error) # append test error to list of errors 
        
        print(f"Epoch {epoch} | Train Error {t_error} | Test Error {test_error}")
        epoch += 1 #increment epoch counter 
    y_predicted_test, _ = fwd_pass(X_test, weights)
    return weights, errors, y_predicted_test