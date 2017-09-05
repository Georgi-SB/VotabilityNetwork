# -*- coding: utf-8 -*-
"""
Neural network class and a test class
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage


class NNetwork(object):

    
    def __init__(self, layer_dims, layer_types, X, Y):
        """Arguments: 
            layer_dims: a list of input layer and hidden layer sizes 
            layer_types: a list of strings - the first element is the input and the 
            type is ignored, the last is the output and it can be sigmoid, tanh,
            relu, lrelu, softmax, the rest can be sigmoid, relu, tanh, lrelu
            X: features in vectorized form - each column is a separate training example
            Y: target variable in vectorized form, each column is a separate training example
            
            Some conventions:
                1. layer_dims, layer_types contains input + hidden layers. counting starts from 0!
                2. num_layers excludes 
            """
        self.num_layers     = len(layer_dims)  #  number of layers in the network. input layer is included
        self.layer_dims     = layer_dims
        self.layer_types    = layer_types
        assert len(layer_dims) == len(layer_types), "invalid layer specs!" 
        self.parameters     = self.initialize_parameters()
        self.caches         = {}    # keys: "Zl" and "Al" for the input and output activation
        self.caches['A0']   = X     # input layer 
        self.grads          = {}    # keeps the calculated gradients of the cost wrt weights. keys "dWl", "dbl"
        self.X              = X
        self.Y              = Y
        self.x_size         = X.shape[0] # dimension of input data
        self.m              = X.shape[1] # size of training data set
        self.writeCaches    = True
        


    def Fit_model(self, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):#lr was 0.009
        """Implements the L-layer neural network
            Arguments:
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

        np.random.seed(1)
        self.parameters     = self.initialize_parameters()
        # keep track of cost
        costs               = []    
    
        # Parameters initialization.
        # parameters = initialize_parameters_deep(layers_dims)
    
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: 
            self.forward_pass(self.X)
            
            # Compute cost.
            L=self.num_layers-1 # number of 
            cost = self.compute_cost(self.caches["A"+str(L)], self.Y)
    
            # Backward propagation.
            self.backward_pass()

            # Update parameters.
            self.update_parameters(learning_rate)
                
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            #if print_cost and i % 100 == 0:
            costs.append(cost)
            
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
        return self.parameters, self.costs
   
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
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) / np.sqrt(self.layer_dims[l-1]) #*0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1])), "Arc weight matrices does not match the NN architecture"
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1)), "Bias vector sizes does not match the NN architecture"
            
        return parameters
    
    def forward_pass(self, X):
        """
            Implements the cross entropy cost function. Currently assumes output has dimension = 1

            Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
        """
        
        self.caches = []
        A = X
    
        for l in range(1, self.num_layers):
            A_prev = A 
            A, Z = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], 
                                                         self.parameters['b' + str(l)], 
                                                         activation = self.layer_types[l])
            if self.writeCaches:
                self.caches['A'+str(l)]= A
                self.caches['Z'+str(l)]= Z
    
        assert(A.shape == self.Y.shape), "Non matching output"  
        return A 
        
   
    def compute_cost(self):
        """
            Implements the cross entropy cost function. Currently assumes output has dimension = 1

            Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
        """
        m = self.Y.shape[1]
        L = self.num_layers-1  # the number of hidden layers
        assert ('A'+str(L) in self.caches), "Cost is not defined as model ouput has not been calculated!" 
        # Compute loss from AL and Y.
        cost = (1./m) * (-np.dot(self.Y,np.log(self.caches['A'+str(L)]).T) - np.dot(1-self.Y, np.log(1-self.caches['A'+str(L)]).T))
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
    
        return cost
    
    def backward_pass(self):
        """
            Implements the backward propagation 
    
            Arguments:
    
                Returns:
            grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
                 """
        grads = {}
        L = self.num_layers-1  # the number of hidden layers
        assert ('A'+str(L) in self.caches), "Fail: Backward propagation is possible only after forward propagation!" 
        AL = self.caches['A'+str(L)]
        self.Y = self.Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        # Initializing the backpropagation
        self.grads["dA" + str(L)] = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
        # self.grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, L, activation = layer_types[L-1])
    
        for l in reversed(range(L)):
            # lth layer:  gradients.
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], l, activation = self.layer_types[l])
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def linear_activation_backward(self, dA, l, activation):
        """
            Implements the backward propagation for the LINEAR->ACTIVATION layer.
    
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
            dZ = self.relu_backward(dA, l)
            dA_prev, dW, db = self.linear_backward(dZ, l)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, l)
            dA_prev, dW, db = self.linear_backward(dZ, l)
    
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

    def sigmoid_backward(self, dA, l):
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
    
    def update_parameters(self,learning_rate):
        """
            Update parameters using gradient descent
    
            Arguments:
            learning rate -- 
    
            Returns:
            parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
        """
    
        L = self.num_layers-1  # the number of hidden layers

        # Update rule for each parameter. 
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * self.grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * self.grads["db" + str(l+1)]
        
        return self.parameters

    def predict(self, Xn, Yn):
        """
            This function is used to predict the results of the  L-layer neural network.
    
            Arguments:
            X -- data set of examples you would like to label
            parameters -- parameters of the trained model
            
            Returns:
                p -- predictions for the given dataset X
            """
        assert Xn.shape[1] == Yn.shape[1], "nb x-examples  does not match nb y examples"
        m = Xn.shape[1] # nb of examples
        assert Xn.shape[0]==self.X.shape[0], "data shape does not match input layer"
        predict = np.zeros((1,m))
    
        # Forward propagation
        original_chaches_mode = self.writeCaches
        self.writeCaches = False
        probs = self.forward_pass(Xn)
        self.writeCaches = original_chaches_mode
    
        # convert probas to 0/1 predictions
        predict[probs<=0.5]=0.0
        predict[probs>0.5]=1.0
    
        #for i in range(0, probs.shape[1]):
        #    if probas[0,i] > 0.5:
        #        predict[0,i] = 1
        #    else:
        #        p[0,i] = 0
    
        #print results
        print ("predictions/true labels: " + str(predict)+" "+str(self.Y))
        print("Accuracy: "  + str(np.sum((predict == self.Y)/m)))
        
        return predict

    
    def linear_activation_forward(self,A_prev, W, b, activation):
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
        
        Z = W.dot(A_prev) + b # np.dot(W,A)+b
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
    
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        return A, Z
    
    
    def sigmoid(self,Z):
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

    def relu(self,Z):
        """
            Implement the RELU function.

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            """
            
        A = np.maximum(0,Z)
        # assert(A.shape == Z.shape)
        return A
    
    def l_relu(self,Z, eps=0.01):
        """
            Implement the leaky RELU function.

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            """
            
        A = np.maximum(eps*Z,Z)
        # assert(A.shape == Z.shape)
        return A
