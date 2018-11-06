import numpy as np
import preprocessing
from constants import IMAGE_WIDTH, GRAY_SCALES

def load_and_process(train_images_file, train_keys_file, test_images_file):
    """
    Load all the datas from the given files and split them in a dictionnary
    to have easy access to all of them.
    :param train_images_file: File path of the training images
    :type train_images_file: string
    :param train_images_file: File path of the training targets
    :type train_images_file: string
    :param test_images_file: File path of the test images
    :type test_images_file: string
    :returns: A dictionnary containing the organized datas
    :rtype: dictionnary
    """
    #Dictionnary containing every dataset
    datasets = {}
    #Load the training targets
    datasets['train_y'] = load_keys(train_keys_file)
    #Load the training datas
    datasets['train_x'] = load_and_process_images(train_images_file)
    #Load the test datas
    datasets['test_x'] = load_and_process_images(test_images_file)
    return datasets

def load_keys(keys_file):
    """
    Load the targets from the given file.
    :param keys_file: File path of the targets
    :type keys_file: string
    :returns: An array containing the targets
    :rtype: 1D numpy array
    """
    #Use the loadtxt function to get the targets, convert the array to the float type
    return (np.loadtxt(keys_file, comments=('#'), dtype='str')[:,1]).astype(np.float)

def load_and_process_images(images_file):
    """
    Load and process the images from the given file.
    :param images_file: File path of the images
    :type images_file: string
    :returns: Array containing the processed images
    :rtype: 2D numpy array
    """
    #Load images, without separations between each
    images_lines = np.loadtxt(images_file, comments=('#','I'))
    #Create a numpy array with the right dimensions
    images = np.full((len(images_lines), int(IMAGE_WIDTH**2)), GRAY_SCALES)
    #Prepare the circular mask to apply to the datas
    circular_mask = preprocessing.create_circular_mask(IMAGE_WIDTH, IMAGE_WIDTH)
    #Separate each image and add them to an array
    pixel_index = 0
    image_index = 0
    for y in range(images_lines.shape[0]):
        if (y % IMAGE_WIDTH == 0 and y > 0):
            pixel_index = 0
            #Automatically rotate the current image
            #Then reshape it in a square
            img_square = preprocessing.auto_rotate(images[image_index]).reshape((IMAGE_WIDTH, IMAGE_WIDTH))
            #Blur the square shaped image
            img_square = preprocessing.blur(img_square)
            #Apply circle mask
            img_square[~circular_mask] = GRAY_SCALES
            #reshape in line again to add it to the datas
            images[image_index] = img_square.reshape(IMAGE_WIDTH**2)
            image_index += 1
        for x in range(images_lines.shape[1]):
            images[image_index][pixel_index] -= images_lines[y][x]
            pixel_index += 1
    return images


