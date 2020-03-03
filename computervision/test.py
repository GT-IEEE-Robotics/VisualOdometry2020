"""
File:          test.py
Author:        Alex Cui
Last Modified:
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from feature_extraction import Feature_Extraction

#this file tests 10 images from the field with feature extraction
if __name__ == '__main__':
    path = '../real_images_input/'
    for i in range(10):
        num = str(i + 1)
        filename = path +'r_' + num + '.png'
        fe = Feature_Extraction(filename)
        fe.denoise_real_image()
        fe.ignore_above()
        fe.detect_features()
        rt = fe.image
        out = 'output_' + num + '.png'
        cv2.imwrite('output/' + out, rt)