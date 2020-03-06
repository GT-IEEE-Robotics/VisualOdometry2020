"""
File:          Feature_Extraction.py
Author:        Alex Cui
Version:       2.0 (added localization methods)
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from feature_extraction import Feature_Extraction

class Feature_Extraction:

    def __init__(self, img):
        """Initializes feature extraction elements by 
        taking in the path of the image to be processed
        """
        self.image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.scale_percent = 20
        self.real_detector = cv2.ORB_create(nfeatures = 10000, scoreType=cv2.ORB_FAST_SCORE)
        self.simulator_detector = cv2.ORB_create(nfeatures=50, scoreType=cv2.ORB_FAST_SCORE)
        self.yellow_lower = np.array([22,60,200],np.uint8)
        self.yellow_upper = np.array([60,255,255],np.uint8)
        self.image_with_features = None
        self.keypoints = None
        self.descriptor = None
        self.good_matches = []
        self.threshold #need to determine this
        self.matching_sim = None

    def denoise_real_image(self):
        """Denoise the image before running feature extraction
        """

        #resize image to the value of scale_percent
        #width = int(self.image.shape[1] * self.scale_percent / 100)
        #height = int(self.image.shape[0] * self.scale_percent / 100)
        #dim = (width, height)
        dim = (500, 500)
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
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        return yellow_mask

    def __get_highest(self, mask):
        """Private method that computes the height above which 
        everything is ignored
        """
        for i in range (mask.shape[0]):
            s = np.sum(mask[i])
            if (s == 255 * mask.shape[1]):
                return i

    def detect_features_for_real(self):
        """Detect features
        """       
        # find keypoints
        kp = self.real_detector.detect(self.image, None)
        kp, des = self.real_detector.compute(self.image, kp)
        self.keypoints = kp
        self.descriptor = des
        k = cv2.drawKeypoints(self.image, kp, None, color=(255,0,0))
        self.image_with_features = k
        
    def detect_features_for_simulator(self):
        """Detect features
        """       
        # find keypoints
        kp = self.simulator_detector.detect(self.image, None)
        kp, des = self.simulator_detector.compute(self.image, kp)
        self.keypoints = kp
        self.descriptor = des
        k = cv2.drawKeypoints(self.image, kp, None, color=(255,0,0))
        self.image_with_features = k
        
    def compare_real_and_simulator(self, simulator):
        simulator.detect_features_for_simulator()
        FLANN_INDEX_LSH = 6
        flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        des_r = self.descriptor
        des_s = simulator.descriptor

        if type(des_r) != np.float32:
            des_r = np.float32(des_r)

        if type(des_s) != np.float32:
            des_s = np.float32(des_s)
    
        matches = flann.knnMatch(des_r, des_s, k=2)

        #Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75

        for m,n in matches:
            if m.distance < ratio_thresh * n.distance:
                self.good_matches.append(m)
    
    def loop_through_5000(self, path_list):
        for i in path_list:
            self.compare_real_and_simulator(Feature_Extraction(i))
            if (len(self.good_matches) / len(self.keypoints)) > self.threshold:
                self.matching_sim = i
                break
        return self.matching_sim




