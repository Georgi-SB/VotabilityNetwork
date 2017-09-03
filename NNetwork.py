# -*- coding: utf-8 -*-
"""
Neural network class and a test class
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py


class NNetwork(object):

    
    def __init__(self, layer_dims, layer_types, X, Y):
        """Arguments: 
            layer_dims: a list of hidden layer sizes starting with the first hidden  layer
            layer_types: a list of strings - the first element is the input and the 
            type is ignored, the last is the output and it can be sigmoid, tanh,
            relu, lrelu, softmax, the rest can be sigmoid, relu, tanh, lrelu
            X: features in vectorized form - each column is a separate training example
            Y: target variable in vectorized form, each column is a separate training example
            
            Some conventions:
                1. layer_dims, layer_types contains only hidden layers. 
                2. num_layers excludes 
            """
        self.num_layers     = len(layer_dims)  #  number of layers in the network. input layer is not counted
        self.layer_dims     = layer_dims
        self.layer_types    = layer_types
        self.parameters     = self.initialize_parameters()
        self.caches         = {}    # keys: "Zl" and "Al" for the input and output activation
        self.caches['A0']   = X     # input layer 
        self.grads          = {}    # keeps the calculated gradients of the cost wrt weights. keys "dWl", "dbl"
        self.X              = X
        self.Y              = Y
        self.x_size         = X.shape[0] # dimension of input data
        self.m              = X.shape[1] # size of training data set
        


    def initialize_parameters(self):
    """
    Arguments:
    layer_dims -- a list containing the dimensions of each layer in the network
    
    Returns:
    parameters -- python dictionary containing  parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
        np.random.seed(1)
        parameters = {}

        for l in range(1, self.num_layers):
            parameters['W' + str(l)] = 
                np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1])), "Arc weight matrices does not match the NN architecture"
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1)), "Bias vector sizes does not match the NN architecture"
            
        return parameters

     def forward_pass(self):
         """
         Implements forward propagation along the neural network. 
         Fills in the input and output activation caches Zl and Al 
    
            Arguments:
        
            Returns:
                A -- last post-activation value
                """

        self.caches = []
        A = self.X
        L = self.num_layers                  # number of layers in the neural network
    
        for l in range(1, L+1):
            A_prev = A 
            A, Z = linear_activation_forward(A_prev, self.parameters['W' + str(l)], 
                                                         self.parameters['b' + str(l)], 
                                                         activation = self.layer_types[l])
            self.caches['A'+str(l)]= A
            self.caches['Z'+str(l)]= Z
    
    
        assert(A.shape == self.Y.shape), "Non matching output"  
            
        return A  
        
   
    def compute_cost(self, AL, Y):
        """
        Implements the cross entropy cost function. Currently assumes output has dimension = 1

        Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)

        Returns:
            cost -- cross-entropy cost
        """
    
        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
    
        return cost
    
    def L_model_backward(AL, Y):
        """
        Implements the backward propagation 
    
        Arguments:
        AL -- output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 and  1)
        caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
        Returns:
            grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
                 """
        grads = {}
        L = self.num_layers  # the number of layers
        m = AL.shape[1]
        self.Y = self.Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        self.grads["dA" + str(L)] = dAL
        # self.grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, L, activation = layer_types[L-1])
    
        for l in reversed(range(L)):
            # lth layer:  gradients.
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], l, activation = layer_types[l])
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

    return grads


    

    def linear_activation_backward(self, dA, l, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
        Arguments:
            dA -- post-activation gradient for current layer l 
            l -- layer 
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
            
            if activation == "relu":
                dZ = relu_backward(dA, l)
                dA_prev, dW, db = linear_backward(dZ, l)
        
            elif activation == "sigmoid":
                dZ = sigmoid_backward(dA, l)
                dA_prev, dW, db = linear_backward(dZ, l)
    
        return dA_prev, dW, db
    
    def linear_backward(self,dZ, l):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
            dZ -- Gradient of the cost with respect to the linear output (of current layer l)
            l -- layer, needed to get  (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
        A_prev, W, b = cache
        A_prev  = self.caches['A'+str(l-1)]
        W       = self.parameters['W'+str(l)]
        b       = self.parameters['b'+str(l)]
    
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
    
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
    
        return dA_prev, dW, db
    
    
    def relu_backward(self,dA, l):
        """
        Implements the backward propagation for a single RELU unit.

        Arguments:
            dA -- post-activation gradient, of any shape
            l -- layer index for the cache (activation input Zl is needed) for computing backward propagation efficiently

        Returns:
            dZ -- Gradient of the cost with respect to Z
        """
    
        Z = self.caches['Z'+str(l)]
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
    
        assert (dZ.shape == Z.shape)
    
        return dZ

    def sigmoid_backward(dA, l):
        """
        Implements the backward propagation for a single SIGMOID unit.

        Arguments:
            dA -- post-activation gradient, of any shape
             l -- layer index for the cache (activation input Zl is needed) for computing backward propagation efficiently

        Returns:
            Z -- Gradient of the cost with respect to Z
        """
    
        Z = self.caches['Z'+str(l)]
    
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
    
        assert (dZ.shape == Z.shape)
    
        return dZ


 
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        
        Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
            A -- the output of the activation function, also called the post-activation value 
            Z -- the activation/layer input stored for computing the backward pass efficiently
             """
        assert (activation == "sigmoid" or activation=="relu") , "Invalid activation string"
        
        Z = W.dot(A) + b # np.dot(W,A)+b
        if activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "relu":
            A = relu(Z)
    
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        return A, Z
       
   def sigmoid(Z):
       """
       Implements the sigmoid activation in numpy
    
        Arguments:
            Z -- numpy array of any shape
    
        Returns:
            A -- output of sigmoid(z), same shape as Z
        """
        
        A = 1/(1+np.exp(-Z))
        # assert(A.shape == Z.shape)
        return A

    def relu(Z):
        """
        Implement the RELU function.

        Arguments:
            Z -- Output of the linear layer, of any shape

        Returns:
            A -- Post-activation parameter, of the same shape as Z
            cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
            """
    
        A = np.maximum(0,Z)
        # assert(A.shape == Z.shape)
        return A

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
        
class TestNNetwork(object):
     def __init__(self, sizes, layer_types):
        """import data and prepare training"""
        self.train_set_x_orig, self.train_set_y_orig, self.test_set_x_orig, 
        self.test_set_y_orig, self.classes = self.load_data()
        
        
        
        
        
    def load_data():
        """load cat vs no cat testing data
        
        Arguments:
        Returns:
            trainX, trainY, testX, testY
            """
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #  train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #  train set labels

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #  test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #  test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
