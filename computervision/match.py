"""
File:          match.py
Author:        Alex Cui
Last Modified:
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from feature_extraction import Feature_Extraction

class Match:
    #Constructor takes in the path of the real image
    def __init__(self, r_path):
        self.real_image = Feature_Extraction(r_path)
        self.weight_list = []
    
    def match(self):
        self.real_image.denoise_real_image()
        self.real_image.ignore_above()
        self.real_image.detect_features_for_real()
        
        kp1 = self.real_image.keypoints
        des1 = self.real_image.descriptor
        if (np.size(des1) == 0):
            cvError(0, "ORB Matcher", "1st image descriptors empty", __FILE__, __LINE__)
            quit()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        for filename in os.listdir('../PF_Images/sim_5000/'):
            base = len(kp1)
            s_path = '../PF_Images/sim_5000/' + filename
            im_s = Feature_Extraction(s_path)
            im_s.detect_features_for_simulator()
            kp2 = im_s.keypoints
            des2 = im_s.descriptor
            if (np.size(des2) == 0):
                cvError(0, "ORB Matcher", "2nd image descriptors empty", __FILE__, __LINE__)
                quit()
            if (np.size(des2) < 2 or np.size(des1) < 2):
                continue
            try:
                matches = flann.knnMatch(des1,des2, k=2)
            except:
                continue
           
            # store all the good matches as per Lowe's ratio test.
            count = 0
            good = []
            for i, pair in enumerate(matches):
                try:
                    m, n = pair
                    if m.distance < 0.7*n.distance:
                        count += 1
                        good.append(m)

                except ValueError:
                    pass
            weight = count / base
            self.weight_list.append(weight)