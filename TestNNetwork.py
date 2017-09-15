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
   def __init__(self, hidden_layer_sizes, hidden_layer_types, learning_rate = 0.0075, num_epochs = 3000
                , print_cost=True):
       """Arguments:
            hidden_layer_sizes -- list of hidden layer sizes
            hidden_layer_types -- list of hidden layer types "relu", "sigmoid"
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
    
            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """
       self.train_x, self.train_y, self.test_x, self.test_y, self.classes = self.load_cat_pics_data()
       self.layer_dims = [self.train_x.shape[0]] + hidden_layer_sizes
       self.layer_types = ["sigmoid"]+hidden_layer_types
       self.learning_rate = learning_rate
       self.num_epochs = num_epochs
       self.print_cost = print_cost

        
        
   def run_test(self):


      network_object = NNetwork.NNetwork(self.layer_dims,
                                         self.layer_types,
                                         self.train_x,
                                         self.train_y,
                                         use_dropout=False,
                                         use_l2_regularization = False)
         
      # network_object.fit_model(batchX = self.train_x[:, 0:100],
      #                         batchY = self.train_y[:, 0:100],
      #                         learning_rate = self.learning_rate,
      #                         num_iterations = self.num_iterations,
      #                         print_cost=self.print_cost)

      network_object.fit_model(X=self.train_x,
                               Y=self.train_y,
                               mini_batch_size=self.train_x.shape[1],
                               optimization_mode="adam",
                               learning_rate= 0.0075,   # self.learning_rate,
                               num_epochs=self.num_epochs,
                               print_cost=self.print_cost)

      print("results on train:")
      network_object.predict(self.train_x, self.train_y)
      print("results on test:")
      network_object.predict(self.test_x, self.test_y)

   def run_gradient_check(self):
      network_object = NNetwork.NNetwork(self.layer_dims,
                                          self.layer_types,
                                          self.train_x[:, 0:8],
                                          self.train_y[:, 0:8],
                                          use_l2_regularization=False)

      network_object.gradient_check(epsilon=1e-7)

   def benchmark_model(self):

      print("BENCHMARKING")

      train_x, train_y, test_x, test_y, classes = self.load_cat_pics_data()

      network_object = NNetwork.NNetwork( [train_x.shape[0],20, 7, 5, 1], ["sigmoid","relu", "relu", "relu", "sigmoid"], train_x, train_y)


      parameters, costs = network_object.fit_model(X = train_x,
                                                   Y = train_y,
                                                   mini_batch_size=train_y.shape[1],
                                                   optimization_mode="gradient_descend",
                                                   learning_rate=self.learning_rate,
                                                   num_epochs=self.num_epochs,
                                                   print_cost=self.print_cost)

      print("results on train:")
      network_object.predict(train_x, train_y)
      print("results on test:")
      network_object.predict(test_x, test_y)
      assert np.abs(costs[0] - 0.771749)<0.000001, "cost is different. step: 0"
      assert np.abs(costs[100] - 0.672053)<0.000001, "cost is different. step: 100"
      assert np.abs(costs[200] - 0.648263)<0.000001, "cost is different. step: 200"
      assert np.abs(costs[300] - 0.611507)<0.000001, "cost is different. step: 300"
      assert np.abs(costs[400] - 0.567047)<0.000001, "cost is different. step: 400"
      assert np.abs(costs[500] - 0.540138)<0.000001, "cost is different. step: 500"
      assert np.abs(costs[600] - 0.527930)<0.000001, "cost is different. step: 600"
      assert np.abs(costs[700] - 0.465477)<0.000001, "cost is different. step: 700"
      assert np.abs(costs[800] - 0.369126)<0.000001, "cost is different. step: 800"
      assert np.abs(costs[900] - 0.391747)<0.000001, "cost is different. step: 900"
      assert np.abs(costs[1000] - 0.315187)<0.000001, "cost is different. step: 1000"
      assert np.abs(costs[1100] - 0.272700)<0.000001, "cost is different. step: 1100"
      assert np.abs(costs[1200] - 0.237419)<0.000001, "cost is different. step: 1200"
      assert np.abs(costs[1300] - 0.199601)<0.000001, "cost is different. step: 1300"
      assert np.abs(costs[1400] - 0.189263)<0.000001, "cost is different. step: 1400"
      assert np.abs(costs[1500] - 0.161189)<0.000001, "cost is different. step: 1500"
      assert np.abs(costs[1600] - 0.148214)<0.000001, "cost is different. step: 1600"
      assert np.abs(costs[1700] - 0.137775)<0.000001, "cost is different. step: 1700"
      assert np.abs(costs[1800] - 0.129740)<0.000001, "cost is different. step: 1800"
      assert np.abs(costs[1900] - 0.121225)<0.000001, "cost is different. step: 1900"
      assert np.abs(costs[2000] - 0.113821)<0.000001, "cost is different. step: 2000"
      assert np.abs(costs[2100] - 0.107839)<0.000001, "cost is different. step: 2100"
      assert np.abs(costs[2200] - 0.102855)<0.000001, "cost is different. step: 2200"
      assert np.abs(costs[2300] - 0.100897)<0.000001, "cost is different. step: 2300"
      assert np.abs(costs[2400] - 0.092878)<0.000001, "cost is different. step: 2400"
      assert np.abs(costs[2500] - 0.088413)<0.000001, "cost is different. step: 2500"
      assert np.abs(costs[2600] - 0.085951)<0.000001, "cost is different. step: 2600"
      assert np.abs(costs[2700] - 0.081681)<0.000001, "cost is different. step: 2700"
      assert np.abs(costs[2800] - 0.078247)<0.000001, "cost is different. step: 2800"
      assert np.abs(costs[2900] - 0.075444)<0.000001, "cost is different. step: 2900"
      print("BENCHMARK PASSED!!!")

   def load_cat_pics_data(self):
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



# For benchmarking the core network run this:
################################################
# test_object = TestNNetwork([20, 7, 5, 1], ["relu","relu","relu","sigmoid"], learning_rate = 0.0075, num_epochs = 3000, print_cost=True)


# test_object.benchmark_model()
################################################


# For gradient check run this:
################################################
#test_object = TestNNetwork([20, 7, 5, 1], ["relu","relu","relu","sigmoid"],
#                           learning_rate = 0.0075,
#                           num_iterations = 3000,
#                           print_cost=True)


#test_object.run_gradient_check()
################################################


test_object = TestNNetwork([20, 7, 5, 1], ["relu","relu","relu","sigmoid"], learning_rate = 0.0075, num_epochs = 3000, print_cost=True)

#test_object = TestNNetwork([20, 7, 5, 1], ["tanh","tanh","tanh","sigmoid"], learning_rate = 0.0075, num_epochs = 3000, print_cost=True)


test_object.run_test()
