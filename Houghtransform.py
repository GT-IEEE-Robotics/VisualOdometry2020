import cv2
import numpy as np
#This function performs a Hough transform on an image and produces an image with recognized lines
#It also tries to remove duplicate lines

#parameters
#+- slope threshold to remove duplicates
slope = 0.03
#+- y-intercept threshold to remove duplicates
intercept = 15

def Htransform(filename, returnedFilename):
    #read in grayscale version of picture
    img = cv2.imread(filename, 0)
    print(img.shape)
    #resized picture
    resize = cv2.resize(img, (960,720))
    print(resize.shape)
    
    edges = cv2.Canny(resize, 50, 100, apertureSize = 3)
    #cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    i = 0
    lineList = []
    for line in lines:
        #print(" ")
        #print("RUN" + str(i))
        rho,theta = line[0]
        #print("rho:" + str(rho))
        #print("theta:" + str(theta))
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        #print("x0:" + str(x0))
        y0 = b * rho
        #print("y0:" + str(y0))
        # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        x1 = int(x0 + 1000 * (-b))
        #print("x1:" + str(x1))
        # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        y1 = int(y0 + 1000 * (a))
        #print("y1:" + str(y1))
        # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        x2 = int(x0 - 1000 * (-b))
        #print("x2:" + str(x2))
        # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        y2 = int(y0 - 1000 * (a))
        #print("y2:" + str(y2))
        m = (y2 - y1) / (x2 - x1)
        #print("m:" + str(m))
        b = y0 - m*(x0)
        #print("b:" + str(b))
        tup = (i, rho, theta, x1, y1, x2, y2, m, b)
        lineList.append(tup)
        i = i + 1
        
    writeList = []
    i = 0;
    for l in lineList:
        notAdded = True
        for w in writeList:
            if (w[0][7] - slope < l[7] < w[0][7] + slope and w[0][8] - intercept < l[8] < w[0][8] + intercept):
                notAdded = False
                w.append(l)
                break
        if (notAdded):
            writeList.append([l])
            
    for w in writeList:
        if (len(w) == 1):
            cv2.line(resize, (w[0][3], w[0][4]), (w[0][5], w[0][6]), (0, 255, 0), 2)
        else:
            x1sum = 0
            y1sum = 0
            x2sum = 0
            y2sum = 0
            size = len(w)
            for t in w:
                x1sum = x1sum + t[3]
                y1sum = y1sum + t[4]
                x2sum = x2sum + t[5]
                y2sum = y2sum + t[6]
            x1avg = int(x1sum / size)
            y1avg = int(y1sum / size)
            x2avg = int(x2sum / size)
            y2avg = int(y2sum / size)
            cv2.line(resize, (x1avg, y1avg), (x2avg, y2avg), (0, 255, 0), 2)
                
    cv2.imwrite(returnedFilename, resize)
    cv2.imshow('image', resize)
    
"""
#tests
Htransform("PF1.jpeg", "PF1line.jpeg")
Htransform("PF2.jpeg", "PF2line.jpeg")
Htransform("PF3.jpeg", "PF3line.jpeg")
"""


