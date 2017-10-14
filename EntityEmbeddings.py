# coding: utf-8

# # Neural Network with TensorFlow
#

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from scipy import ndimage
from tensorflow.python.framework import ops


class NeuralNetworkTf(object):
    def __init__(self, triplets,  l2_regularization=False, l2_lambda=0.01):
        """Arguments:
            triplets: a list of input layer and hidden layer sizes
            layer_types: a list of strings - the first element is the input and the
            type is ignored, the last is the output and it can be sigmoid, tanh,
            relu, lrelu, softmax, the rest can be sigmoid, relu, tanh, lrelu
            X: features in vectorized form - each column is a separate training example. numpy array
            Y: target variable in vectorized form, each column is a separate training example. numpy array

            Some conventions:
                1. layer_dims, layer_types contains input + hidden layers. counting starts from 0!
                2. num_layers excludes
            """
        np.random.seed(1)
        self.entity_alphabet = []
        self.relation_alphabet = []
        self.parameters = {}
        # element i,j,k is 1 if the triplet (entity_alphabet[i], elation_alphabet[j], entity_alphabet[k])

        self.sparse_triplet_tensor = []



    def get_alphabets(self):
        pass


    def fit_model(self, X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
                  num_epochs=1500, minibatch_size=32, print_cost=True, optimizer="adam"):
        """
        Implements the calibration of the   tensorflow neural network model defined in the object/class

        Arguments:
        X_train -- training set, of shape (input size , number of training examples )
        Y_train -- test set, of shape (output size , number of training examples )
        X_test -- training set, of shape (input size , number of training examples )
        Y_test -- test set, of shape (output size, number of test examples)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        tf.reset_default_graph()
        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)  # to keep consistent results
        seed = 3  # to keep consistent results
        (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]  # n_y : output size
        assert n_x == self.layer_dims[0], "inconsistent input layer dimension and training set! "
        assert n_y == self.layer_dims[-1], "inconsistent output layer dimension and training set! "
        costs = []  # keep track of the cost

        # Create tensorflow Placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = self.initialize_tf_trainable_parameters()

        # Forward propagation: the forward propagation in the tensorflow graph
        Z_final, A_final = self.forward_propagation(X, parameters)

        # Cost function:  add cost function to tensorflow graph
        cost = self.compute_softmax_cross_entropy_cost(Z_final, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer
        if optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimizer == "gradient_descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        elif optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):

                epoch_cost = 0.  # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size)  # number of minibatches  in the train set
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    dummy, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost for every 100-th epoch
                if print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # put the  parameters in a variable
            self.parameters = sess.run(parameters)
            print("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(A_final), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

            return self.parameters

    def create_placeholders(self, n_x, n_y):
        """
            Creates the placeholders for the tensorflow session.

            Arguments:
            n_x -- scalar, size of an image/input vector
            n_y -- scalar, number of classes

            Returns:
            X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
            Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

            Misc:
            -  use None because it let's us be flexible on the number of examples we will use for the placeholders.
              In fact, the number of examples during test/train is different.
            """
        X = tf.placeholder(tf.float32, shape=[n_x, None])
        Y = tf.placeholder(tf.float32, shape=[n_y, None])
        return X, Y

    def initialize_tf_trainable_parameters(self):
        """
            Initializes parameters to build a neural network with tensorflow.

            Returns:
            parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        tf.set_random_seed(1)

        self.parameters = {}

        for l in range(1, self.num_layers):
            if self.l2_regularization:
                self.parameters['W' + str(l)] = tf.get_variable('W' + str(l),
                                                                [self.layer_dims[l], self.layer_dims[l - 1]]
                                                                ,
                                                                initializer=tf.contrib.layers.xavier_initializer(seed=1)
                                                                , regularizer=tf.contrib.layers.l2_regularizer(
                        self.l2_lambda))
            else:
                self.parameters['W' + str(l)] = tf.get_variable('W' + str(l),
                                                                [self.layer_dims[l], self.layer_dims[l - 1]],
                                                                initializer=tf.contrib.layers.xavier_initializer(
                                                                    seed=1))

            self.parameters['b' + str(l)] = tf.get_variable('b' + str(l), [self.layer_dims[l], 1]
                                                            , initializer=tf.zeros_initializer())

        self.are_params_initialized = True

        return self.parameters

    def forward_propagation(self, X, write_caches=True):
        """
        Implements the forward propagation for the model:

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing the  parameters "W1", "b1", "W2", "b2", etc


        Returns:
        fills in the cache and returns the input and output activation of the last layer
        """
        if not self.are_params_initialized:
            self.initialize_tf_trainable_parameters()

        A = X
        Z = X
        for l in range(1, self.num_layers):
            Z = tf.add(tf.matmul(self.parameters['W' + str(l)], A), self.parameters['b' + str(l)])
            if self.layer_types[l] == "relu":
                A = tf.nn.relu(Z)
            elif self.layer_types[l] == "sigmoid":
                A = tf.nn.sigmoid(Z)
            elif self.layer_types[l] == "tanh":
                A = tf.nn.tanh(Z)
            elif self.layer_types[l] == "softmax":
                A = tf.transpose(tf.nn.softmax(logits=tf.transpose(Z)))
            elif self.layer_types[l] == "selu":
                selu_alpha = tf.constant(1.6732632423543772848170429916717, name="selu_alpha")
                selu_scale = tf.constant(1.0507009873554804934193349852946, name="selu_scale")
                A = tf.multiply(selu_scale, tf.where(Z >= 0.0, Z, tf.add(tf.multiply(selu_alpha, tf.exp(Z)),
                                                                         tf.multiply(-1.0, selu_alpha))))
            else:
                print("Unknown activation function! Using selu")
            # write input activation Z and output activation A of layer l into the cache
            if write_caches:
                self.caches["Z" + str(l)] = Z
                self.caches["A" + str(l)] = A

        return Z, A

    def predict_index(self, X):

        x = tf.placeholder("float", [None, None])

        z, a = self.forward_propagation(x, write_caches=False)
        p = tf.argmax(z)

        sess = tf.Session()
        prediction = sess.run(p, feed_dict={x: X})
        sess.close()
        return prediction

    def compute_softmax_cross_entropy_cost(self, Z, Y):
        """
        Computes the softmax based cross-entropy cost

        Arguments:
        Z -- input of the last, softmax, layer after the full  forward propagation (output of the last LINEAR unit)
        Y -- "true" labels vector placeholder, same shape as Z

        Returns:
        cost - Tensor of the cost function
        """
        # l2 regularization
        l2_regularizer = tf.nn.l2_loss(self.parameters["W1"])
        if self.l2_regularization:
            for l in range(2, self.num_layers):
                l2_regularizer += tf.nn.l2_loss(self.parameters["W" + str(l)])

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        # we work with dimensions: (size of the layer , size of the mini-batch) while tf is (size of the mini-batch , size of the layer)
        # see https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
        logits = tf.transpose(Z)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) +
                              0.5 * self.l2_lambda * l2_regularizer)

        return cost

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 , 1 ), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        #  Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        #  Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in the partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    # Y = tf.Variable(tf.add(tf.matmul(W,X),b), name='Y')
    Y = tf.add(tf.matmul(W, X), b, name='Y')
    # init = tf.global_variables_initializer()
    ### END CODE HERE ###

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate

    ### START CODE HERE ###
    sess = tf.Session()
    # sess.run(init)
    result = sess.run(Y)
    ### END CODE HERE ###

    # close the session
    sess.close()

    return result


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Change the index below and run the cell to visualize some examples in the dataset.

# Example of a picture
index = 12
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# flatten the image dataset, then normalize it by dividing by 255.
# Convert each label to a one-hot vector

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.


# Convert training and test labels to one hot matrices


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

tf_nn = NeuralNetworkTf([12288, 25, 12, 6], ["sigmoid", "selu", "selu", "softmax"], use_dropout=False,
                        l2_regularization=True, l2_lambda=0.07)

tf_nn.fit_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, learning_rate=0.0001,
                num_epochs=1500, minibatch_size=32, print_cost=True, optimizer="adam")

## PUT YOUR IMAGE NAME
my_image = "duos.jpg"


# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
my_image_prediction = tf_nn.predict_index(my_image)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
