import numpy as np
from .neuron import Neuron
from utils.constants import IMAGE_WIDTH, GRAY_SCALES

class Neural_network:
    """
    Class that instanciates a neural network containing one output layer.
    """

    def __init__(self, learning_rate = 0.05, training_proportion = 0.66):
        """
        Initializes a neural network, setting its learning rate and the used training
        proportion.
        :param learning_rate: set the learning rate of the neural network (default: 0.05)
        :type learning_rate: float
        :param training_proportion: set the proportion between the training set and the
        test set (default: 0.66)
        :type training_proportion: float
        """
        self.output_layer = []
        self.learning_rate = learning_rate
        self.training_proportion = training_proportion

    def fit(self, X, y, max_epochs = 100, min_epochs = 15, threshold = 0.90):
        """
        Fit the neural network to data matrix X and target(s) y.
        :param X: Matrix containing the datas to train on
        :type X: 2D python array
        :param y: Array containing the targets corresponding to the datas in X
        :type y: 1D python array
        :param max_epochs: Maximum number of epochs (one forward pass and one 
        backward pass of the entire training set) that the training process will do
        (default: 100)
        :type max_epochs: integer
        :param min_epochs: Minimum number of epochs to do (default: 15)
        :type min_epochs: integer
        :param threshold: Success rate threshold value, stops the process if the network
        evaluation returns more than that value (default: 0.94)
        :type threshold: float
        """
        #Creation of the output layer
        for i in range(len(set(y))):
            #Creation of one neuron, not setting any bias value
            neuron = Neuron()
            #Initialization of the weights of that neuron
            neuron.weights = [0 for j in range(len(X[0]))]
            #Add the neuron to the output layer
            self.output_layer.append(neuron)
        #training and learning process for a number of epochs between min_epochs and max_epochs
        for epoch in range(max_epochs):
            #Random separation of the datas between the training and the testing sets
            train_x, train_y, test_x, test_y = self.split_in_sets(X, y)
            #Train for one epoch
            success_rate = self.train_epoch(train_x, train_y, test_x, test_y, epoch)
            #If the network answer to the user expectations
            #the training process is stopped
            if epoch >= min_epochs and success_rate >= threshold:
                break

    def train_epoch(self, train_x, train_y, test_x, test_y, epoch):
        """
        Train the network for one epoch.
        :param train_x: The training datas
        :type train_x: 2D numpy array
        :param train_y: The training targets
        :type train_y: 1D python array
        :param test_x: The test datas
        :type test_x: 2D numpy array
        :param test_y: The test targets
        :type test_y: 1D python array
        :param epoch: Number of the epoch to display to the user
        :type epoch: integer
        :returns: The success rate of the evaluation of the epoch
        :rtype: float
        """
        #Training and learning process for one epoch
        for index in range(len(train_y)):
            self.learn(train_x[index], train_y[index])
        #Evaluate the updated network with the testing datas
        success_rate = self.evaluate(test_x, test_y)
        #Display informations to the user
        print('# epoch {0}: success rate {1}%'.format(epoch, success_rate * 100))
        return success_rate

    def evaluate(self, test_x, test_y):
        """
        Feed forward a set of test datas in the network to evaluate it in 
        its actual state and returns the computed success rate.
        :param test_x: The test datas
        :type test_x: 2D numpy array
        :param test_y: The test targets
        :type test_y: 1D python array
        :returns: The computed success rate
        :rtype: float
        """
        #Initializes the success counter
        success_count = 0.0
        for index in range(len(test_y)):
            #If prediction is good the counter is incremented by 1
            if self.predict(test_x[index]) == test_y[index]:
                success_count += 1.0
        #Compute the succes rate
        return success_count / len(test_y)

    def predict(self, x):
        """
        Feed forward one data x in the neural networks and returns the
        network predicted target.
        :param x: One data to classify
        :type x: 1D numpy array
        :returns: Prediction of the network
        :rtype: integer
        """
        #Initializes the max_output to the first output neuron computed output
        max_output = self.output_layer[0].compute_output(x)
        #Initializes the max_index to the index of the first output neuron
        max_index = 0
        #Goes through the other output neurons
        for i in range(1, len(self.output_layer)):
            #Memoize the computed output of the neuron
            output = self.output_layer[i].compute_output(x)
            #If the output is greater than all the ones previously computed
            #Memoize it and memoize the index of the corresponding neuron
            if output > max_output:
                max_output = output
                max_index = i
        #Compute the predicted target by adding one to the neuron index
        return max_index + 1

    def learn(self, x, y):
        """
        Feed forward a data x in the network and analyzes the network output to
        update the neurons weights to better fit the data.
        :param x: The data to fit the network to
        :type x: 1D numpy array
        :param y: The target to fit the network to
        :type y: integer
        """
        #Update the weights for every output neuron
        for neuron_index in range(len(self.output_layer)):
            #Compute the desired output for the selected neuron
            desired_output = (neuron_index + 1 == y)
            #Feed forward the data and compute the selected neuron output
            neuron_output = self.output_layer[neuron_index].compute_output(x)
            #Compute the error between the desired output and the neuron output
            error = desired_output - neuron_output
            #Update every weights of the selected neuron
            for weight_index in range(len(self.output_layer[neuron_index].weights)):
                #Compute the delta to add to the selected weight
                delta = self.learning_rate * error * self.normalize_input(x[weight_index])
                self.output_layer[neuron_index].weights[weight_index] += delta
    
    def normalize_input(self, pixel):
        """
        Normalize the given pixel value between 0 and 1.
        :param pixel: Pixel value
        :type pixel: integer
        :returns: Normalized value
        :rtype: float
        """
        return pixel / GRAY_SCALES

    def split_in_sets(self, X, y):
        """
        Split the given datas X and targets y into random training and test sets,
        respecting the set self.training_proportion value.
        :param X: The datas to split
        :type X: 2D python array
        :param y: The targets to split
        :type y: 1D python array
        :returns train_x: The training set of datas
        :rtype train_x: 2D numpy array
        :returns train_y: The training set of targets
        :rtype train_y: 1D python array
        :returns test_x: The test set of datas
        :rtype test_x: 2D numpy array
        :returns test_y: The test set of targets
        :rtype test_y: 1D python array
        """
        #Create a permutation of the indices from 0 to the number of datas - 1
        permutation = np.random.permutation(len(y))
        #Initializes the train_x variable to a 2D numpy array respecting the training proportion
        train_x = np.zeros(shape=(int(len(y) * self.training_proportion), int(IMAGE_WIDTH**2)))
        #Initializes the test_x variable to a 2D numpy array respecting the training proportion
        test_x = np.zeros(shape=(int(len(y) * (1 - self.training_proportion)), int(IMAGE_WIDTH**2)))
        #Initialized the train_x and test_y variables to 1D python arrays 
        train_y, test_y = ([] for i in range(2))
        #Add datas to training set or test set respecting the training proportion
        index = 0
        for perm in permutation:
            if index < len(y) * self.training_proportion:
                train_x[index] = X[perm]
                train_y.append(y[perm])
            else:
                test_x[index - int(len(y) * self.training_proportion)] = X[perm]
                test_y.append(y[perm])
            index += 1
        return train_x, train_y, test_x, test_y

