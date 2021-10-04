import cv2
import numpy as np

def process(imageName):
    output = None
    # check the format of image
    if imageName.endswith('jpg') or imageName.endswith('png') or imageName.endswith('jpeg'):
        # read the image in grayscale
        # grayscaleImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE) / 255
        grayscaleImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

        output = cv2.resize(grayscaleImage, (80, 80))

    return output


def process_test(imageName, row, cox):
    output = None
    # check the format of image
    # if imageName.endswith('jpg') or imageName.endswith('png') or imageName.endswith('jpeg'):
        # read the image in grayscale
    image = cv2.imread(imageName)/ 255

    output = cv2.resize(image, (row, cox))

    return output