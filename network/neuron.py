import numpy as np
from utils.constants import GRAY_SCALES

class Neuron:
    """
    Neuron class, instanciates a neuron object.
    """

    def __init__(self, bias = 0):
        """
        Initializes a neuron, setting its bias.
        :param bias: Bias value of the neuron (default: 0)
        :type bias: float
        """
        self.weights = []
        self.bias = bias

    def compute_output(self, image):
        """
        Compute the neuron output for a given image.
        :param image: The image to classify
        :type image: 1D numpy array
        :returns: The computed neuron output
        :rtype: float
        """
        #Initializes the neuron output to the bias value of the neuron
        output = self.bias
        #For every pixel of the image add the computed value to the output variable
        for index in range(len(image)):
            #Normalize the pixel between 0 and 1
            normalized_input = image[index] / GRAY_SCALES
            #Multiply the nomalized pixel value to the corresponding weight value
            #and add it to the final output
            output += normalized_input * self.weights[index]
        #Put the output in the activation function (here a sigmoid) to compute
        #the final output of the neuron
        return 1 / (1 + np.exp(-output))
