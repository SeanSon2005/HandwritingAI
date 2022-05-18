#import tensorflow as tf
#from tensorflow import keras
from dis import dis
import numpy as np
import cv2
import os
import re

lower_threshold = np.array([0,0,100])
upper_threshold = np.array([170,45,255])
lower_threshold2 = np.array([85,0,0])
upper_threshold2 = np.array([107,255,255])

def annotate_image(image):
    print("Generating Legibility Report")
    scale_percent =  30# percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    mask2 = cv2.inRange(hsv, lower_threshold2, upper_threshold2)
    img = mask + mask2

    coord_array = generate_coordinates(img,mask2)
    #for i in coord_array:
        #x = int(i[2])
        #y = int(i[4])
        #cv2.circle(mask,(x,y),5,(0,255,0),2)

    cv2.imshow("image",coord_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def generate_coordinates(image,mask):
    image = image[:,120:(image.shape[1]-1)]
    width = image.shape[1]
    height = image.shape[0]
    output = cv2.blur(image,(int(width/150),int(height/150)))
    output = cv2.inRange(output,0,225)
    lines = find_lines(mask)
    for i in range(len(lines)-1):
        sub_img = output[lines[i]:lines[i+1]]
        cv2.imshow("image",sub_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for j in range(width):
            if np.sum(sub_img[:,j]) > 60:
                print("1",end="")
            else:
                print("0",end="")
        print("")

    return output
def find_lines(image):
    width = image.shape[1]
    height = image.shape[0]
    output = cv2.blur(image,(int(width/100),int(height/100)))
    output = cv2.inRange(output,0,50)
    lines = []
    cur_line = False
    iter = 1
    for i in output:
        line_likely = (np.sum(i) < 30000)
        if(line_likely and not cur_line):
            cur_line = True
            lines.append(iter + (int(width/200)))
        elif(not line_likely):
            cur_line = False
        iter+=1
    return lines

os.chdir(r"C:\Users\astro\Documents\Handwriting\Images")
img = cv2.imread("image0.jpg", cv2.IMREAD_COLOR)
annotate_image(img)