#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import cv2
import os
import re
import math

def annotate_image(image):
    print("Generating Legibility Report")
    scale_percent =  20# percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    hsv =  cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    no_lines = cv2.inRange(hsv,(70,0,0),(112,255,255))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    img = cv2.addWeighted(img,1.8,img,-0.5,0)
    brighten = 75
    for i in range(no_lines.shape[0]):
        for j in range(no_lines.shape[1]):
            if no_lines[i][j] == 255:
                if img[i][j] + brighten > 255:
                    img[i][j] = 255
                else:
                    img[i][j] += brighten
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(resized, contours, -1, (0,255,0), 1)

    cv2.imshow("image",resized)
    cv2.imshow("contrast",img)
    cv2.imshow("oh",no_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


os.chdir(r"C:\Users\astro\Documents\Handwriting\Images")
img = cv2.imread("image0.jpg", cv2.IMREAD_COLOR)
annotate_image(img)