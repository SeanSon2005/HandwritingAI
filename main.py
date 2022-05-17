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
        camera()
        annotate_image()

def create_practice():
    global practice_length
    global practice_text
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
    camera()
    annotate_image()

def annotate_image():
    global image
    print("Generating Legibility Report")
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_array = generate_images(grayImage)
    for i in image_array:
        image = i[0]

def generate_images(image):
    return [(image,0,0)]

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



