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
import copy
import math


class NNetwork(object):

    def __init__(self, layer_dims, layer_types,   use_dropout = False, use_l2_regularization = False):
        """Arguments: 
            layer_dims: a list of input layer and hidden layer sizes 
            layer_types: a list of strings - the first element is the input and the 
            type is ignored, the last is the output and it can be sigmoid, tanh,
            relu, lrelu, softmax, the rest can be sigmoid, relu, tanh, lrelu
            X: features in vectorized form - each column is a separate training example. numpy array
            Y: target variable in vectorized form, each column is a separate training example. numpy array
            
            Some conventions:
                1. layer_dims, layer_types contains input + hidden layers. counting starts from 0!
                2. num_layers excludes 
            """
        self.num_layers     = len(layer_dims)  #  number of layers in the network. input layer is included
        self.layer_dims     = layer_dims
        self.layer_types    = layer_types
        assert len(layer_dims) == len(layer_types), "invalid layer specs!" 

        # parameters and caches
        self.parameters     = {} # keeps the parameters of the nn. keys "bl" and "Wl"
        self.caches         = {}    # keys: "Zl" and "Al" for the input and output activation
        self.grads          = {}    # keeps the calculated gradients of the cost wrt weights. keys "dWl", "dbl"
        self.writeCaches    = True

        # momentum and gradient variance
        self.momentum = {} # keeps the momentum for each parameter. keys "bl" and "Wl"
        self.variance = {} # keeps the gradient second moment  for each parameter. keys "bl" and "Wl"
        self.beta_momentum = 0.9
        self.beta_variance = 0.999

        # the leaky realu constant
        self.l_relu_epsilon = 0.01

        # grad check is only for debugging - very slow
        self.run_grad_check = False

        # drop out parameters
        self.use_dropout    = use_dropout
        self.keep_probs     = np.ones(self.num_layers)*0.8
        self.keep_probs[0]  = 1.0
        self.keep_probs[self.num_layers-1] = 1.0

        # L2 regularization
        self.l2_reg         = use_l2_regularization
        self.lambd          = 10.0

    def fit_model(self, X, Y, mini_batch_size = 128, optimization_mode = "adam", learning_rate = 0.0075, num_epochs = 3000, print_cost=True, seed = 1):
        """Implements the L-layer neural network training
            Arguments:
            X, Y - training dataset
            mini_batch_size - size of the mini-batch
            optimization_mode - "adam" , "gradient_descend", "momentum", "nesterov_momentum", "rmsprop", "nadam"
            learning_rate -- learning rate of the gradient descent update rule
            num_epochs -- number of epochs of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

        tic = time.process_time()
        # Parameters initialization.
        self.parameters, self.momentum, self.variance = self.initialize_parameters()
        # keep track of cost
        costs = []
        np.random.seed(seed)

        m = X.shape[1]
        parameter_update_counter = 0
        num_minibatches = math.floor(m / mini_batch_size) +1
        # Loop (gradient descent)
        for i in range(num_epochs):
            #increment seed to achhieve different reshuffle
            seed = seed + 1
            # get the minibatches
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)
            minibatch_counter = 0
            minibatch_average_cost = 0.0
            for minibatch in minibatches:
                minibatch_counter = minibatch_counter + 1
                parameter_update_counter = parameter_update_counter + 1
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Forward propagation:
                self.forward_pass(minibatch_X)

                # Compute cost.
                L = self.num_layers - 1  # number of
                # use only if sigmoid explodes
                #if self.use_dropout:
                self.caches["A" + str(L)] = np.minimum(np.maximum(self.caches["A" + str(L)], 0.00000001), 0.99999999)
                cost = self.compute_cost(self.caches["A" + str(L)], minibatch_Y)
                # Backward propagation.
                self.grads = self.backward_pass(minibatch_Y)

                # Update parameters.
                self.update_parameters(self.grads, parameter_update_counter, learning_rate, optimization_mode)

                # Print the cost every 100 training example and every minibatch
                minibatch_average_cost += cost / num_minibatches
                #if print_cost and i % 1000 == 0:
                #    print("Cost after epoch %i, minibatch %i: %f" % (i, minibatch_counter, cost))
            if print_cost and i % 100 == 0:
                print("Cost after epoch %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
           # if print_cost and i % 1000 == 0:
        #    print("Average cost after epoch %i: %f" % (i, minibatch_average_cost))
                # if print_cost and i % 100 == 0:

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        toc = time.process_time()
        print("Network trained in " + str(1000 * (toc - tic)) + "ms")
        return self.parameters, costs

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
        momentum = {}
        variance = {}

        for l in range(1, self.num_layers):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2.0/self.layer_dims[l-1]) #*0.01    !!! use this for benchmark!!! *np.sqrt(1.0/self.layer_dims[l-1])
            #He initialization
            # parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) np.sqrt(2)/ np.sqrt((self.layer_dims[l-1] + self.layer_dims[l]) #*0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            #initialize momentum and variance
            momentum['W' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
            momentum['b' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
            variance['W' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
            variance['b' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

            #assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1])), "Arc weight matrices does not match the NN architecture"
            #assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1)), "Bias vector sizes does not match the NN architecture"

        return parameters, momentum, variance

    
    def forward_pass(self, batchX):
        """
            Implements the forward propagation

            Arguments:
            X -- input features

            Returns:
            model output -- output from the last hidden layer
        """
        
        # self.caches = {}
        self.caches['A0'] = batchX
        A = batchX

        for l in range(1, self.num_layers):
            A_prev = A
            A, Z = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)],
                                                  self.parameters['b' + str(l)],
                                                  activation = self.layer_types[l])
            if self.use_dropout and l < (self.num_layers-1): # never drop out neurons from the last layer
                D = np.random.binomial(1, self.keep_probs[l], A.shape)
                #D = np.random.rand(A.shape[0], A.shape[1])
                #D = (D <= self.keep_probs[l])
                A = np.multiply(A, D)
                A = A/self.keep_probs[l]
                self.caches['D' + str(l)] = D

            if self.writeCaches:
                self.caches['A'+str(l)]= A
                self.caches['Z'+str(l)]= Z
    
        # assert(A.shape == self.Y.shape), "Non matching output"
        return A 
        
   
    def compute_cost(self, model_output, y):
        """
            Implements the cross entropy cost function. Currently assumes output has dimension = 1

            Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
        """
        m = y.shape[1]
        assert model_output.shape == y.shape, "dimensions of model output and true labels do not match"
        # L = self.num_layers-1  # the number of hidden layers
        # assert ('A'+str(L) in self.caches), "Cost is not defined as model ouput has not been calculated!" 
        # Compute loss from AL and Y.
        #cost = (1./m) * (-np.dot(y, np.log(model_output.T)) - np.dot(1-y, np.log(1-model_output.T)))
        if self.layer_types[self.num_layers-1] != "softmax":
            logprobs = np.multiply(-np.log(model_output), y) + np.multiply(-np.log(1 - model_output), 1 - y)
            cost = 1. / m * np.sum(logprobs)
        else:
            logprobs = np.sum( np.multiply(-np.log(model_output), y), axis=0, keepdims=True)
            cost = np.sum(logprobs, axis=1)/m

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
    
        return cost

    def compute_regularized_cost(self, model_output, y, lambd):
        """
            Implements the cross entropy cost function with L2 regularization. Currently assumes output has dimension = 1

            Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)
            lambd -- regularization constant

            Returns:
            cost -- cross-entropy cost
        """
        cross_entropy_cost = self.compute_cost(model_output=model_output,y=y)
        m = y.shape[1]
        l2_reg_cost = 0.0
        if self.l2_reg:
            for l in range(self.num_layers):
                l2_reg_cost += np.sum(np.square(self.parameters["W"+str(l)]))*lambd/(2.0*m)

        cost = cross_entropy_cost + l2_reg_cost
        return cost

    def backward_pass(self, Y):
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
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        if (self.layer_types[L] != "softmax"): # if last layer is not softmax
            # Initializing the backpropagation
            grads["dA" + str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            # self.grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, L, activation = layer_types[L-1])
    
            for l in reversed(range(L)):
                # lth layer:  gradients.
                dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], l+1, activation = self.layer_types[l+1])
                if self.use_dropout and l>0:
                    dA_prev_temp = np.multiply(dA_prev_temp, self.caches["D"+str(l)])/self.keep_probs[l]
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
        else: # last layer is softmax
            # Initializing the backpropagation
            grads["dA" + str(L)] = -np.sum(np.divide(Y, AL), axis=0, keepdims=True)
            grads["dZ" + str(L)] = self.caches["A"+ str(L)] - Y  # gradient of softmax wrt last input activation ZL
            grads["dA" + str(L - 1)] , grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(grads["dZ" + str(L)], L)

            if self.use_dropout and L > 1:
                grads["dA" + str(L - 1)] = np.multiply(grads["dA" + str(L - 1)], self.caches["D" + str(L-1)]) / self.keep_probs[L-1]

            for l in reversed(range(L-1)):
                # lth layer:  gradients.
                dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], l+1, activation = self.layer_types[l+1])
                if self.use_dropout and l>0:
                    dA_prev_temp = np.multiply(dA_prev_temp, self.caches["D"+str(l)])/self.keep_probs[l]
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp

        return grads


    def linear_activation_backward(self, dA, layer, activation):
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
            dZ = self.relu_backward(dA, layer)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, layer)
        elif activation == "l_relu":
            dZ = self.l_relu_backward(dA, layer)
        elif activation == "tanh":
            dZ = self.tanh_backward(dA, layer)
        elif activation == "selu":
            dZ = self.selu_backward(dA, layer)

        if self.use_dropout and (layer != 0) and (layer != (self.num_layers-1)):
            dZ = np.multiply(dZ, self.caches["D"+str(layer)])


        dA_prev, dW, db = self.linear_backward(dZ, layer)
    
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
        if self.l2_reg:
            dW = dW + W * self.lambd/m
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
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    
        # When z <= 0,   set dz to 0 as well.
        dZ[Z <= 0] = 0
    
        assert (dZ.shape == Z.shape)
    
        return dZ

    def l_relu_backward(self, dA, l, eps=0.01):
        """
            Implements the backward propagation for a single leaky RELU unit.

            Arguments:
            dA -- post-activation gradient, of any shape
            l -- layer index for the cache (activation input Zl is needed) for computing backward propagation efficiently

            Returns:
            dZ -- Gradient of the cost with respect to Z
        """
        eps = self.l_relu_epsilon
        Z = self.caches['Z' + str(l)]
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        q = np.ones(dZ.shape)
        q[Z <= 0] = eps
        # When z <= 0,   set dz to epsilon .
        dZ = np.multiply(dZ, q)

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
        #s = self.caches['A'+str(l)]

        dZ = dA * s * (1-s)
    
        assert (dZ.shape == Z.shape)
    
        return dZ

    def selu_backward(self, dA, l):
        """
            Implements the backward propagation for a single SELU unit.

            Arguments:
            dA -- post-activation gradient, of any shape
             l -- layer index for the cache (activation input Zl is needed) for computing backward propagation efficiently

            Returns:
            Z -- Gradient of the cost with respect to Z
        """
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        Z = self.caches['Z' + str(l)]
        Al = self.caches['A' + str(l)]
        return np.multiply(dA,  np.where(Z >= 0.0, scale*alpha , Al + scale*alpha))


    def tanh_backward(self, dA, l):
        """
            Implements the backward propagation for a single SIGMOID unit.

            Arguments:
            dA -- post-activation gradient, of any shape
             l -- layer index for the cache (activation input Zl is needed) for computing backward propagation efficiently

            Returns:
            Z -- Gradient of the cost with respect to Z
        """

        #Z = self.caches['Z' + str(l)]
        s = self.caches['A' + str(l)]

        #dZ = np.multiply(dA,1.0-np.multiply(s , s))
        return  np.multiply(dA, 1.0-np.multiply(s, s))

        # assert (dZ.shape == Z.shape)



    def update_parameters(self,gradients, parameter_update_counter, learning_rate, optimization_mode = "adam", epsilon = 1e-8):
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
            #update momentum
            self.momentum["W" + str(l + 1)] = self.beta_momentum * self.momentum["W" + str(l + 1)] + (1 - self.beta_momentum) * gradients["dW" + str(l+1)]
            self.momentum["b" + str(l + 1)] = self.beta_momentum * self.momentum["b" + str(l + 1)] + (1 - self.beta_momentum) * gradients['db' + str(l+1)]
            #update variance
            self.variance["W" + str(l + 1)] = self.beta_variance * self.variance["W" + str(l + 1)] + (1 - self.beta_variance) * np.multiply(gradients["dW" + str(l + 1)], gradients["dW" + str(l + 1)])
            self.variance["b" + str(l + 1)] = self.beta_variance * self.variance["b" + str(l + 1)] + (1 - self.beta_variance) * np.multiply(gradients['db' + str(l + 1)],gradients['db' + str(l + 1)])
            # get bias correction terms
            momentum_correction = 1.0 - np.power(self.beta_momentum, parameter_update_counter)
            variance_correction = 1.0 - np.power(self.beta_variance, parameter_update_counter)


            addOnW = gradients["dW" + str(l+1)]
            addOnb = gradients["db" + str(l+1)]

            if optimization_mode == "momentum":
                addOnW =  self.momentum["W" + str(l + 1)]
                addOnb =  self.momentum["b" + str(l + 1)]
            elif optimization_mode == "adam":
                addOnW =  np.divide(self.momentum["W" + str(l + 1)]/momentum_correction, epsilon + np.sqrt(self.variance["W" + str(l + 1)]/variance_correction))
                addOnb =  np.divide(self.momentum["b" + str(l + 1)]/momentum_correction, epsilon + np.sqrt(self.variance["b" + str(l + 1)]/variance_correction))
            elif optimization_mode == "rmsprop":
                addOnW =  np.divide(gradients["dW" + str(l+1)],epsilon + np.sqrt(self.variance["W" + str(l + 1)]/variance_correction ))
                addOnb =  np.divide(gradients["db" + str(l+1)],epsilon + np.sqrt(self.variance["b" + str(l + 1)]/variance_correction  ))

            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * addOnW
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * addOnb

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
        # assert Xn.shape[0]==self.X.shape[0], "data shape does not match input layer"
        predict = np.zeros(Yn.shape)
    
        # Forward propagation
        original_caches_mode = self.writeCaches
        self.writeCaches = False
        probs = self.forward_pass(Xn)
        self.writeCaches = original_caches_mode
        if probs.shape[0]>1: #softmax last layer
            probs = probs-np.max(probs, axis=0, keepdims=True)
            predict[probs>=0.0] = 1.0
            accuracy = 1.0 - np.sum((predict != Yn))/(2*m)
        else:
            # convert probabilities to 0/1 predictions
            predict[probs <= 0.5] = 0.0
            predict[probs > 0.5] = 1.0
            accuracy = np.sum((predict == Yn))/m
    
        #for i in range(0, probs.shape[1]):
        #    if probas[0,i] > 0.5:
        #        predict[0,i] = 1
        #    else:
        #        p[0,i] = 0
    
        #print results
        print ("predictions/true labels: " + str(predict)+" "+str(Yn))
        print("Accuracy: " + str(accuracy))
        
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
        assert (activation == "softmax" or activation == "sigmoid" or activation=="relu" or activation=="selu"or activation=="tanh" or activation=="l_relu"or activation=="l_relu") , "Invalid activation string"

        Z = W.dot(A_prev) + b # np.dot(W,A)+b
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
        elif activation == "l_relu":
            A = self.l_relu(Z=Z)
        elif activation == "tanh":
            A = self.tanh_np(Z=Z)
        elif activation == "selu":
            A = self.selu(Z=Z)
        elif activation == "softmax":
            A = self.softmax(Z=Z)
    
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        assert (Z.shape == A.shape)
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

    def tanh_np(self, Z):
        """
                    Implements the tanh activation in numpy

                    Arguments:
                    Z -- numpy array of any shape

                    Returns:
                    A -- output of tanh(z), same shape as Z
                """

        A = np.tanh(Z)
        # assert(A.shape == Z.shape)
        return A

    def relu(self, Z):
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
    
    def l_relu(self, Z):
        """
            Implement the leaky RELU function.

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            """
        eps = self.l_relu_epsilon
        A = np.maximum(eps*Z, Z)
        # assert(A.shape == Z.shape)
        return A

    def selu(self, Z):
        """
            Implement the self normalizing activation,  SELU function.

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            """


        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(Z >= 0.0, Z, alpha * np.exp(Z) - alpha)


    def softmax(self, Z):
        """
            Implement the softmax final layer   .

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            """
        max_element = np.max(Z,axis=0, keepdims=True)
        e_Z = np.exp(Z - max_element)
        return e_Z / np.sum(e_Z, axis=0, keepdims=True)


    def set_keep_probs(self, keep_probs):
        """
                    keep_probs for dropout setter

                    Arguments:
                    keep_probs -- numpy array (1, num_layers) with the probs to keep the neurons
                    first and last entry are irrelevent and are set to 1

                    Returns:
                    A -- Post-activation parameter, of the same shape as Z
                    """
        assert keep_probs.shape == (1, self.num_layers) , "Invalid keep_probs dimension"
        self.keep_probs = keep_probs
        self.keep_probs[0]=1.0
        self.keep_probs[self.num_layers-1] = 1.0
        for a in self.keep_probs:
            assert ( (a>0.0) and (a <= 1.0) ), "Invalid keep_probs value. Must be strictly greater than 0 and smaller or equal than 1.0"

    def random_mini_batches(self, X, Y, mini_batch_size=128, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1  / 0 ), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        np.random.seed(seed)
        m = X.shape[1]  # number of training examples
        mini_batches = []

        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in the partitionning

        if num_complete_minibatches<2:
            mini_batches.append((X,Y))
        else:
            #  Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((1, m))

            #  Partition (shuffled_X, shuffled_Y). Minus the end case.

            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # Handling the end case (last mini-batch < mini_batch_size)
            if m % mini_batch_size != 0:
                mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
                mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

        return mini_batches

    def gradient_check(self,  batchX, batchY, epsilon = 1e-7):
        """ Implements a gradient checking routine to verify the backprop implementation"""

        # make a full forward and backward pass
        self.parameters = self.initialize_parameters()

        # Forward propagation:
        self.forward_pass(batchX)

        # Compute cost.
        L = self.num_layers - 1  # number of
        cost = self.compute_cost(self.caches["A" + str(L)], batchY)
        # Backward propagation.
        self.grads = self.backward_pass(batchY)

        #calculate numerical gradient approximation
        for l in range(L):
            parameters_plus  = copy.deepcopy(self.parameters)
            J_plus = np.zeros(parameters_plus["W"+str(l+1)].shape)
            J_minus = np.zeros(J_plus.shape)
            gradapprox = np.zeros(J_plus.shape)
            parameters_minus = copy.deepcopy(self.parameters)
            for i in range(self.layer_dims[l+1]):
                for j in range(self.layer_dims[l]):
                    #bump values
                    parameters_plus["W" + str(l + 1)][i][j] += epsilon
                    parameters_minus["W" + str(l + 1)][i][j] -= epsilon

                    A_plus = self.forward_pass_for_gradient_check(batchX, parameters_plus)
                    J_plus = self.compute_cost(A_plus, self.Y )
                    A_minus = self.forward_pass_for_gradient_check(batchX, parameters_minus)
                    J_minus = self.compute_cost(A_minus, batchY)
                    gradapprox[i][j]=(J_plus - J_minus)/(2.0*epsilon)
                    # restore values
                    parameters_plus["W" + str(l + 1)][i][j] -= epsilon
                    parameters_minus["W" + str(l + 1)][i][j] += epsilon


            numerator = np.linalg.norm(gradapprox - self.grads["dW"+str(l+1)])  # Step 1'
            denominator = np.linalg.norm(gradapprox) + np.linalg.norm(self.grads["dW"+str(l+1)])  # Step 2'
            difference = numerator / denominator  # Step 3'
            ### END CODE HERE ###

            print("Checking W gradients for hidden layer " + str(l+1))
            if difference > 1e-7:
                print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                    difference) + "\033[0m")
            else:
                print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                    difference) + "\033[0m")

            parameters_plus = copy.deepcopy(self.parameters)
            J_plus = np.zeros(parameters_plus["b" + str(l + 1)].shape)
            J_minus = np.zeros(J_plus.shape)
            gradapprox = np.zeros(J_plus.shape)
            parameters_minus = copy.deepcopy(self.parameters)
            for i in range(self.layer_dims[l + 1]):
                # bump values
                parameters_plus["b" + str(l + 1)][i,0] += epsilon
                parameters_minus["b" + str(l + 1)][i,0] -= epsilon
                A_plus = self.forward_pass_for_gradient_check(batchX, parameters_plus)
                J_plus = self.compute_cost(A_plus, self.Y)
                A_minus = self.forward_pass_for_gradient_check(batchX, parameters_minus)
                J_minus = self.compute_cost(A_minus, batchY)
                gradapprox[i][0] = (J_plus - J_minus) / (2.0 * epsilon)
                # restore values
                parameters_plus["b" + str(l + 1)][i,0] -= epsilon
                parameters_minus["b" + str(l + 1)][i,0] += epsilon

            numerator = np.linalg.norm(gradapprox - self.grads["db"+str(l+1)])  # Step 1'
            denominator = np.linalg.norm(gradapprox) + np.linalg.norm(self.grads["db"+str(l+1)])  # Step 2'
            difference = numerator / denominator  # Step 3'
            ### END CODE HERE ###

            print("Checking b gradients for hidden layer " + str(l + 1))
            if difference > 1e-7:
                print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                    difference) + "\033[0m")
            else:
                print("\033[92m" + "The backward propagation works perfectly fine! difference = " + str(
                    difference) + "\033[0m")

    def forward_pass_for_gradient_check(self, x, parameters):
        """
            Implements the cross entropy cost function. Currently assumes output has dimension = 1

            Arguments:
            AL -- probability vector corresponding to the label predictions, shape (1, number of examples)
            Y -- true "label" vector ( 0 or 1), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
        """

        # self.caches = {}
        A = x

        for l in range(1, self.num_layers):
            A_prev = A
            A, _ = self.linear_activation_forward(A_prev, parameters['W' + str(l)],
                                                  parameters['b' + str(l)],
                                                  activation=self.layer_types[l])

        return A


    @staticmethod
    def  normalize_input_data(train_x, test_x):
        """
            Normalize the input data

                Arguments:
                X -- input features as np.array

                Returns:
                model output -- output from the last hidden layer
        """
        size = train_x.shape[1]
        mean = np.sum(train_x, axis=1, keepdims=True) / size
        train_x = train_x - mean
        test_x = test_x - mean
        stdev = np.sqrt(np.sum(np.multiply(train_x, train_x), axis=1, keepdims=True) / (size - 1))
        train_x = np.divide(train_x, stdev)
        test_x = np.divide(test_x, stdev)
        return train_x, test_x