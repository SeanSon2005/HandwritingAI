import numpy as np
import cv2
import os
import kivy
from kivy.app import App
from kivy.uix.label import Label


image = np.array([0])
type_conv = 0
image_dir = r"C:\Users\astro\Documents\Handwriting\Images"

class Practice_App(App):
    def build(self):
        return Label(text ="Hello World !")

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
    
    print("Please choose a setting to change:")
    print("1. Update Image Path (Current:",image_dir,")")
    if (type_conv == 0):
        print("2. Type of Interpretation (Current: text )")
    else:
        print("2. Type of Interpretation (Current: math )")
    change = input()
    print("------------------")
    if(int(change) == 1):
        image_dir = input("Type new image path")
    else:
        print("Enter 1 for text mode, 2 for math mode")
        mode = input()
        type_conv = int(mode)
    print("\n------------------")
    print("Changes Saved")

def generate_practice():
    app = Practice_App
    app.run()

print("Welcome to Handwrite")
while True:
    print("Please choose an option:")
    print("1 Use Camera")
    print("2 Generate practice")
    print("3 Create practice")
    print("4 Take Test")
    print("5 Generate Annotated PDF")
    print("6 Change Practice Settings")
    print("7 Quit APP")
    decision = input()
    print("------------------")
    if(int(decision) == 1):
        print("Opening Camera...")
        camera()
    if(int(decision) == 2):
        print("Generating Practice...")
        generate_practice()
    if(int(decision) == 3):
        print("Opening Practice Creator...")
    if(int(decision) == 4):
        print("Generating Test...")
    if(int(decision) == 5):
        print("Generating Annotated PDF...")
    if(int(decision) == 6):
        print("Opening Settings...")
        settings()
    if(int(decision) == 7):
        print("Closing App...")
        break
    print("")
    print("------------------")



