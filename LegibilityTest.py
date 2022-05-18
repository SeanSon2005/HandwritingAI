#import tensorflow as tf
#from tensorflow import keras
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
    for i in coord_array:
        cv2.rectangle(resized,(i[0],i[1]),(i[2],i[3]),(0,255,0),2)


    cv2.imshow("image",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def generate_coordinates(image,mask):
    image = image[:,120:(image.shape[1]-10)]
    width = image.shape[1]
    height = image.shape[0]
    output = cv2.blur(image,(int(width/150),int(height/150)))
    output = cv2.inRange(output,0,225)
    lines = find_lines(mask)
    letter_cells = []
    for i in range(len(lines)-1):
        sub_img = output[lines[i]:lines[i+1]]
        iter = 0
        start = 0
        while iter < (width-1):
            start = iter
            if np.sum(sub_img[:,iter]) > 60:
                while np.sum(sub_img[:,iter]) > 60 and (iter < (width-1)):
                    iter+=1
                letter_cells.append((start+120,lines[i],iter+120,lines[i+1]))
            iter+=1

    return letter_cells
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