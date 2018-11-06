import sys
import matplotlib.pyplot as plt
import numpy as np
from network import Neural_network
from utils import data_loader
from utils.constants import IMAGE_WIDTH

def classify_datas(net, X):
    """
    Classify the datas X using the trained neural network net, print 
    the predicted values using the key-file format.
    :param net: The trained neural network
    :type net: Neural_network
    :param X: The datas to classify
    :type X: 2D python array
    """
    #Initializes the images_count variable
    images_count = len(X) / IMAGE_WIDTH
    #Temporary variable that will convert the given datas
    #to the right format for the net
    test_x = np.zeros(shape=(1, int(IMAGE_WIDTH**2)))
    #For every data gives it to the net to predict the classification
    for index in range(images_count):
        #Convert data to the right format
        test_x[0] = X[index]
        #Give it to the net and get the prediction
        prediction = net.predict(test_x[0])
        #Print the made prediction in the key-file format
        print('Image{0} {1}'.format(index + 1, prediction))

if __name__ == '__main__':
    #Get the files paths from the user
    training_images = sys.argv[1]
    training_keys = sys.argv[2]
    test_images = sys.argv[3]
    
    #Load the datas from the given files paths
    datasets = data_loader.load_and_process(training_images, training_keys, test_images)
    
    #Create the net
    net = Neural_network()
    #Fit the net to the datas
    net.fit(datasets['train_x'], datasets['train_y'])
    #Predict the results for the test datas
    classify_datas(net, datasets['test_x'])
    
