import numpy as np
from constants import IMAGE_WIDTH, HALF_IMAGE

def create_circular_mask(h, w):
    """
    Creates a circular mask to apply to the images
    :param h: Height of the image
    :type h: integer
    :param w: Width of the image
    :type w: integer
    :returns: The created mask
    :rtype: Boolean 2D numpy array
    """
    #Use the middle of the image
    center = [int(w / 2), int(h / 2)]
    #Use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    #Compute the distance from the center for every pixel of the image
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    #Create a mask of booleans depending of if the pixel is in the circle or not
    return dist_from_center <= radius

def auto_rotate(img):
    """
    Automatically rotate the given image to put the eyes on top
    :param img: The image to rotate
    :type img: 1D numpy array, shape: (1, IMAGE_WIDTH**2) 
    :returns: The rotated image
    :rtype: 2D numpy array, shape: (IMAGE_WIDTH, IMAGE_WIDTH)
    """
    #Rotate the image in all possible angles
    #Memoize the image and the corresponding sum of the top half of the image in an array
    #After each rotation reshape the image to rotate in the shape: (1, IMAGE_WIDTH**2)
    halves = [img[:HALF_IMAGE].sum()]
    img_2 = np.rot90(img.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), k=1).reshape(IMAGE_WIDTH**2)
    halves.append(img_2[:HALF_IMAGE].sum())
    img_3 = np.rot90(img_2.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), k=1).reshape(IMAGE_WIDTH**2)
    halves.append(img_3[:HALF_IMAGE].sum())
    img_4 = np.rot90(img_3.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), k=1).reshape(IMAGE_WIDTH**2)
    halves.append(img_4[:HALF_IMAGE].sum())
    index = np.argmin(halves)
    #Return the image that have the smallest first half, 
    #meaning that the eyes should be on top
    if index == 1:
        return img_2
    elif index == 2:
        return img_3
    elif index == 3:
        return img_4
    return img

def get_blur_value(img, i, j):
    """
    Compute the blurred value of the pixel.
    :param img: The image to blur
    :type: 2D numpy array, shape :(IMAGE_WIDTH, IMAGE_WIDTH)
    :param i: The row index of the pixel
    :type i: integer
    :param j: The column index of the pixel
    :type j: integer
    :returns: The blurred value
    :rtype: integer
    """
    #Initializes the mask_sum variable to 0
    mask_sum = 0
    #Initializes the changes_count variable to 1, 
    #because the first count has no conditions
    changes_count = 1
    #Add the value of the selected pixel to the mask_sum variable
    mask_sum += img[i][j]
    #If conditions are true add the values of the pixels around the
    #selected pixel and increment the count by 1 each time
    if i > 0 and j > 0:
        mask_sum += img[i - 1][j - 1]
        changes_count += 1
    if j > 0:
        mask_sum += img[i][j - 1]
        changes_count += 1
    if i < IMAGE_WIDTH - 1 and j > 0:
        mask_sum += img[i + 1][j - 1]
        changes_count += 1
    if i > 0:
        mask_sum += img[i - 1][j]
        changes_count += 1
    if i < IMAGE_WIDTH - 1:
        mask_sum += img[i + 1][j]
        changes_count += 1
    if i > 0 and j < IMAGE_WIDTH - 1:
        mask_sum += img[i - 1][j + 1]
        changes_count += 1
    if j < IMAGE_WIDTH - 1:
        mask_sum += img[i][j + 1]
        changes_count += 1
    if i < IMAGE_WIDTH - 1 and j < IMAGE_WIDTH - 1:
        mask_sum += img[i + 1][j + 1]
        changes_count += 1
    #Compute the blurred value, depending on how many pixels have been
    #added to the count, 9 at most, 4 at least
    return mask_sum / changes_count

def blur(img):
    """
    Blur the given image.
    :param img: The image to blur
    :type: 2D numpy array, shape :(IMAGE_WIDTH, IMAGE_WIDTH)
    :returns: The blurred image
    :rtype: 2D numpy array, shape :(IMAGE_WIDTH, IMAGE_WIDTH)
    """
    #Goes through every pixel of the image and change into its new blurred value
    for pixel_row in range(IMAGE_WIDTH):
        for pixel_col in range(IMAGE_WIDTH):
            #Change the pixel into its computed blurred value
            img[pixel_row][pixel_col] = get_blur_value(img, pixel_row, pixel_col)
    return img

