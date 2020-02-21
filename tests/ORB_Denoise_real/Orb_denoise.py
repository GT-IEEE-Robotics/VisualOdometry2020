import numpy as np
import cv2
from matplotlib import pyplot as plt

orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

for i in range(10):
    num = str(i + 1)
    filename_i = 'r_' + num + '.png'   
    img = cv2.imread('../real_images_input/' + filename_i, 1)

    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 7, 21, 50, 10)

    # find keypoints
    kp = orb.detect(dst, None)
    kp, des = orb.compute(dst, kp)

    # plot keypoints
    img2 = cv2.drawKeypoints(dst, kp, None, color=(0,0,255))
    filename = 'ORB_Denoise_r_' + num + '.png'
    cv2.imwrite('Orb_denoise_real_output/' + filename,img2)