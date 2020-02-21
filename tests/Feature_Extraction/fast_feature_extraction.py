import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create FAST object with default values
fast = cv2.FastFeatureDetector_create()

# Plot features on images from the simulator
for i in range(10):
    num = str(i + 1)
    filename_i = 's_' + num + '.png'
    img = cv2.imread('./simulated_images_input/' + filename_i,1)
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
    filename_o = 'FAST_s_' + num + '.png'
    cv2.imwrite('./FAST_output/FAST_simulated_output/' + filename_o,img2)

# Plot features on images from playing field
for i in range(10):
    num = str(i + 1)
    filename_i = 'r_' + num + '.png'   
    img = cv2.imread('./real_images_input/' + filename_i,1)
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
    filename = 'FAST_r_' + num + '.png'
    cv2.imwrite('./FAST_output/FAST_real_output/' + filename,img2)