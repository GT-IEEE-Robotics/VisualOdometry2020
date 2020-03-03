"""
File:          track.py
Author:        Alex Cui
Last Modified:
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from feature_extraction import Feature_Extraction

class Track:

    def __init__(self, img, previous_track):
        """Initializes feature extraction elements by 
        taking in the path of the image to be processed
        """
        self.image = cv2.imread(img)
        self.keypoints = {}
        self.descriptor = {}
        self.detector = cv2.ORB_create()
        self.track = []
        self.previous_track = previous_track

    def update_track(self):
        dt = self.detector
        keypoints = dt.detect(self.image, None)
        keypoints, descriptor = dt.compute(self.image, keypoints)

        countk = 0
        countd = 0
        kp = {}
        des = {}
        for i in keypoints:
            coordinate = i.pt
            kp[countk] = coordinate
            countk += 1
        self.keypoints = kp

        for j in descriptor:
            des[countd] = j
            countd += 1
        self.descriptor = des

        if self.previous_track == None:
            self.track = kp.keys()
        else:
            # find keypoints for current track
            kp1 = np.array(list(self.previous_track.keypoints.keys()))
            des1 = np.array(list(self.previous_track.descriptor.values()))

            # find keypoints for previous track
            kp2 = np.array(list(self.keypoints.keys()))
            des2 = np.array(list(self.descriptor.values()))

            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2, k=2)

            #-- Filter matches using the Lowe's ratio test
            ratio_thresh = 0.7
            good_matches = []
            for m,n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
                    
            self.track.append(len(good_matches))

            
            # For each match...
            for mat in good_matches:

                # Get the matching keypoints for each of the current image
                img_idx = mat.trainIdx
                c = self.keypoints[img_idx]
                l = list(self.keypoints.keys())
                for k in l:
                    if self.keypoints[k] == c:
                        self.track.append(k)
                        break
                        
                self.track.sort()


    def __check_threshold(self):
        pass











