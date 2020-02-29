"""
File:          Feature_Extraction.py
Author:        Alex Cui
Last Modified:
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

class Feature_Extraction:
    """Maintains and coordinates the game loop"""

    def __init__(self, img):
        """Initializes feature extraction elements by 
        taking in the path of the image to be processed
        """
        self.image = cv2.imread(img)
        self.scale_percent = 20
        self.detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        self.yellow_lower = np.array([22,60,200],np.uint8)
        self.yellow_upper = np.array([60,255,255],np.uint8)

    def denoise_real_image(self):
        """Denoise the image before running feature extraction
        """

        #resize image to the value of scale_percent
        width = int(self.image.shape[1] * self.scale_percent / 100)
        height = int(self.image.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

        #Denoise the image
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 7, 21, 50, 10)

    def ignore_above(self):
        """Ignore noises outside the field
        """ 

        mask = self.__compute_mask()
        x = self.__get_highest(mask)
        mask[x:, :] = 255
        mask[:x, :] = 0
        self.image = cv2.bitwise_and(self.image, self.image, mask = mask)

    def __compute_mask(self):
        """Private method that computes mask
        """

        #Create yellow_mask
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        return yellow_mask

    def __get_highest(self, mask):
        """Private method that computes the height above which 
        everything is ignored
        """
        for i in range (mask.shape[0]):
            s = np.sum(mask[i])
            if (s >= 255 * mask.shape[1] / 2):
                return i

    def detect_features(self):
        """Detect features
        """       
        # find keypoints
        kp = self.detector.detect(self.image, None)
        kp, des = self.detector.compute(self.image, kp)
        k = cv2.drawKeypoints(self.image, kp, None, color=(0,0,255))
        self.image = k




