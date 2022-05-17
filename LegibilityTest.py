#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import cv2
import os
import pytesseract
from pytesseract import pytesseract
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

lower_threshold = np.array([0,0,95])
upper_threshold = np.array([150,150,255])

def annotate_image(image):
    print("Generating Legibility Report")
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)

    coord_array = generate_coordinates(mask)
    #cv2.imshow("image",mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(coord_array)

def generate_coordinates(image):
    data = pytesseract.image_to_boxes(image)
    return data


os.chdir(r"C:\Users\astro\Documents\Handwriting\Images")
img = cv2.imread("image0.jpg", cv2.IMREAD_COLOR)
annotate_image(img)