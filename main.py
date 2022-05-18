import numpy as np
import cv2
import os
from random import randint
import kivy
kivy.require('2.1.0') 
from kivy.app import App
from kivy.uix.label import Label
import tensorflow as tf
from tensorflow import keras
import re

image = np.array([0])
type_conv = 0
practice_length = 15
practice_text = []
image_dir = r"C:\Users\astro\Documents\Handwriting\Images"
sys_dir = r"C:\Users\astro\Documents\Handwriting\Sys"
pdf_dir = r"C:\Users\astro\Documents\Handwriting\PDF"

class MyApp(App):
    def build(self):
        texts = ""
        for i in practice_text:
            texts += i
        return Label(text=texts)

def camera():
    global image
    vid = cv2.VideoCapture(0)
    while True:
         ret, frame = vid.read()
         cv2.imshow("normal",frame)
         image = np.copy(frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            file_count = len([name for name in os.listdir('.') if os.path.isfile(name)])
            os.chdir(image_dir)
            complete_path = image_dir + "\image"+str(file_count)+".jpg"
            cv2.imwrite(complete_path, frame)
            print("Saved Image")
            break
    vid.release()
    cv2.destroyAllWindows()
    return frame

def settings():
    global image_dir
    global type_conv
    global practice_length
    
    print("Please choose a setting to change:")
    print("1. Update Image Path (Current:",image_dir,")")
    if (type_conv == 0):
        print("2. Type of Interpretation (Current: english )")
    else:
        print("2. Type of Interpretation (Current: symbols )")
    print("3. Change Practice Length")
    change = input()
    print("------------------")
    if(int(change) == 1):
        image_dir = input("Type new image path")
    elif (int(change) == 2):
        print("Enter 1 for text mode, 2 for math mode")
        mode = input()
        type_conv = int(mode)
    else:
        print("Current Length is " + str(practice_length) + ".")
        print("New Length: ",end="")
        new_length = input()
        practice_length = int(new_length)

    print("\n------------------")
    print("Changes Saved")

def generate_practice(usage):
    global practice_length
    global practice_text
    f = open(sys_dir+"\dictionary.txt", "r")
    dict = f.readlines()
    
    for i in range(practice_length):
        practice_text.append(dict[randint(0, 45382)][:-1])
    if(usage):
        print("------------------")
        print("")
        for i in practice_text:
            print(i,end=" ")
        print("\n")
        print("------------------")
        print("Enter when done")
        done = input()
        print("Launching Camera...")
        image = camera()
        annotate_image(image)

def create_practice():
    global practice_length
    global practice_text
    global image
    print("Please type how many words to input")
    print("Length:",end="")
    length = input()
    practice_length = int(length)
    practice_text = []
    print("Type words")
    for i in range(practice_length):
        text = input()
        practice_text.append(text)
    print("Launching Camera...")
    image = camera()
    annotate_image(image)

def annotate_image(image):
    lower_threshold = np.array([0,0,100])
    upper_threshold = np.array([170,45,255])
    lower_threshold2 = np.array([85,0,0])
    upper_threshold2 = np.array([107,255,255])
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

def take_test():
    generate_practice(False)
    MyApp().run()
    camera()
    annotate_image()

print("Welcome to Handwrite")
while True:
    print("Please choose an option:")
    print("1 Use Camera")
    print("2 Generate practice")
    print("3 Create practice")
    print("4 Take Test")
    print("5 Change Practice Settings")
    print("6 Quit APP")
    decision = input()
    print("------------------")
    if(int(decision) == 1):
        print("Opening Camera...")
        camera()
    if(int(decision) == 2):
        print("Generating Practice...")
        generate_practice(True)
    if(int(decision) == 3):
        print("Opening Practice Creator...")
        create_practice()
    if(int(decision) == 4):
        print("Generating Test...")
        take_test()
    if(int(decision) == 5):
        print("Opening Settings...")
        settings()
    if(int(decision) == 6):
        print("Closing App...")
        break
    print("")
    print("------------------")



