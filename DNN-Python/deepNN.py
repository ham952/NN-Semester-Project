##########################################################################################################################
# Deep Neural Network for Image Classification
# This is a python implementation by Hamza Rafique, during Masters Studies in IAA , Air University, Isb 2019 email : ham952@hotmail.com
# Implmentation is inspired from "Neural Networks and Deep Learning" by "deeplearning.ai"
# Please feel free to use. Dont forget to acknowledge :)
##########################################################################################################################

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from deepNN_utils import *
import argparse
import logging
logging.basicConfig(level=logging.INFO)

np.random.seed(1)

def data_preprocessing():
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    #Example of a picture
    #index = 10
    #plt.imshow(train_x_orig[index])
    #print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.


    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    print ("")

    return train_x, train_y,test_x, test_y

# Two-layer neural network

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".

        A1,cache1 = linear_activation_forward(X,W1,b1,activation='relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,activation='sigmoid')

        # Compute cost
        cost =  compute_cost(A2,Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2,cache2,activation='sigmoid')
        dA0 , dW1, db1 = linear_activation_backward(dA1,cache1,activation='relu')

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# L-layer Neural Network

### CONSTANTS ###
#layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
#layers_dims = [12288, 28, 18, 9, 4, 1]



def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    if learning_rate is None:
        learning_rate = 0.009
    if layers_dims is None:
        layers_dims =  [12288, 28, 6, 1]
        
    print ("Model Dimension :  " , layers_dims)
    print ("Learning rate :  ",learning_rate)
    np.random.seed(1)
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL,caches = L_model_forward(X, parameters)

        # Compute Cost
        cost = compute_cost(AL,Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update Parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters,costs


def main():

    parser = argparse.ArgumentParser(description='Constrructing Deep Neural Network for classification')
    parser.add_argument('--layers_dims',
                        required = False,
                        nargs='+',
                        type=int,
                        metavar='12288, 20, 7, 5, 1',
                        help="layers dimension")
    parser.add_argument('--learning_rate',
                        required=False,
                        type = float,
                        default = 0.009,
                        metavar='0.009',
                        help="Learning rate for model")
    parser.add_argument('--num_iterations',
                        required=False,
                        type = int,
                        default = 2500,
                        metavar='2500',
                        help="Number of iterations for the main loop")

    args = parser.parse_args()

    

    #parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    print("\nINFO : --- Dataset Preprocessing ---\n")
    train_x, train_y,test_x, test_y = data_preprocessing()
    logging.info("Started Model Training\n")
    parameters,costs = L_layer_model(train_x, train_y,args.layers_dims,args.learning_rate, args.num_iterations, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    logging.info("Successfully Completed Training")

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Cost Vs num of iterations plot " )
    plt.show()
    
if __name__ == '__main__':
    main()







