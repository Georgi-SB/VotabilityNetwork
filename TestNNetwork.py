# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import NNetwork 
       
class TestNNetwork(object):
     def __init__(self, hidden_layer_sizes, hidden_layer_types, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):
        """Arguments:
            hidden_layer_sizes -- list of hidden layer sizes
            hidden_layer_types -- list of hidden layer types "relu", "sigmoid"
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
    
            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """
        self.train_x, self.train_y, self.test_x, self.test_y, self.classes = self.load_data()
        self.layer_dims = [self.train_x.shape[0]] + hidden_layer_sizes
        self.layer_types = ["sigmoid"]+hidden_layer_types
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        
        
     def run_test(self):
         
         network_object = NNetwork.NNetwork(self.layer_dims, self.layer_types, self.train_x, self.train_y)
         
         network_object.fit_model(learning_rate = self.learning_rate, num_iterations = self.num_iterations, print_cost=self.print_cost)
         
         network_object.predict(self.test_x, self.test_y)
        
        
        
        
     def load_data(self):
        """load cat vs no cat testing data
            Arguments:
                Returns:
            trainX, trainY, testX, testY
            """
    
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        train_x_orig = np.array(train_dataset["train_set_x"][:]) #  train set features
        train_y = np.array(train_dataset["train_set_y"][:]) #  train set labels

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_x_orig = np.array(test_dataset["test_set_x"][:]) #  test set features
        test_y = np.array(test_dataset["test_set_y"][:]) #  test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
        train_y = train_y.reshape((1, train_y.shape[0]))
        test_y = test_y.reshape((1, test_y.shape[0]))
        
        # Reshape the training and test examples : RGB comes into a single vector
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.
    
        return train_x, train_y, test_x, test_y, classes

test_object = TestNNetwork([20, 7, 5, 1], ["relu","relu","relu","sigmoid"], learning_rate = 0.0075, num_iterations = 3000, print_cost=True)

test_object.run_test()