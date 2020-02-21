import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create ORB object with default values
orb = cv2.ORB_create()

# Create BRIEF extractor
#brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Plot features on images from the simulator
for i in range(10):
    num = str(i + 1)
    filename_i = 's_' + num + '.png'
    img = cv2.imread('../simulated_images_input/' + filename_i,1)

    # find keypoints
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    # plot keypoints
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
    filename_o = 'ORB_s_' + num + '.png'
    cv2.imwrite('./ORB_output/ORB_simulated_output/' + filename_o,img2)

# Plot features on images from playing field
for i in range(10):
    num = str(i + 1)
    filename_i = 'r_' + num + '.png'   
    img = cv2.imread('../real_images_input/' + filename_i,1)

    # find keypoints
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    # plot keypoints
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
    filename = 'ORB_BRIEF_r_' + num + '.png'
    cv2.imwrite('./ORB_output/ORB_real_output/' + filename,img2)





