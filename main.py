import numpy as np
import cv2
import os
from random import randint
from PIL import Image, ImageEnhance
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

type_conv = 0
practice_length = 15
practice_text = []
image_dir = r"C:\Users\astro\Documents\Handwriting\Images"
sys_dir = r"C:\Users\astro\Documents\Handwriting\Sys"
pdf_dir = r"C:\Users\astro\Documents\Handwriting\PDF"

def camera():
    vid = cv2.VideoCapture(0)
    while True:
         ret, frame = vid.read()
         cv2.imshow("normal",frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            file_count = len([name for name in os.listdir('.') if os.path.isfile(name)])
            os.chdir(image_dir)
            complete_path = image_dir + "\image"+str(file_count)+".jpg"
            cv2.imwrite(complete_path, frame)
            print("Saved Image")
            break
    vid.release()
    cv2.destroyAllWindows()
    return frame.copy()

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
    print("Generating Legibility Report")
    result = ImageProcess(image, 1500)
    cv2.imshow("res",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    max_score = findCharacters(result.copy())
    boxes = pytesseract.image_to_string(result.copy())
    count = len(boxes)
    score = count/max_score * 100
    print(score)

def findCharacters(image):
    gray = image.copy()
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    character_count = 0
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        roi = image[y:y + h, x:x + w]

        area = w*h

        if 100 < area < 1500 and (w / h) < 3:
            character_count+=1
            #rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return character_count



def ImageProcess(image, factor_value):
    current_img_width = image.shape[1]
    scale_percent = 100
    if(current_img_width > factor_value):
        scale_percent = factor_value/current_img_width*100
    
    width = int(current_img_width * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cropped = transformation(resized)
    width = cropped.shape[1]
    height = cropped.shape[0]
    cropped = cropped[5:(height-5)]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = BrightnessContrast(3,gray)
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 1)
    result = cv2.bitwise_or(gray, erode)
    return result

def BrightnessContrast(factor, image):
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(img)
    output = enhancer.enhance(factor)
    return np.asarray(img)
    
def blur_and_threshold(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 2)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
    return threshold

def biggest_contour(contours, min_area):
    biggest = None
    max_area = 0
    biggest_n = 0
    approx_contour = None
    for n, i in enumerate(contours):
        area = cv2.contourArea(i)

        if area > min_area / 10:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                biggest_n = n
                approx_contour = approx

    return biggest_n, approx_contour


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def transformation(image):
    image = image.copy()
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.size
    threshold = blur_and_threshold(gray)
    threshold = cv2.blur(threshold,(4,4))
    edges = cv2.Canny(threshold, 50, 150, apertureSize=7)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,0.001 * cv2.arcLength(hull, True), True))
    simplified_contours = np.array(simplified_contours,dtype=object)

    biggest_n, approx_contour = biggest_contour(simplified_contours, image_size)

    threshold = cv2.drawContours(image, simplified_contours, biggest_n, (0, 255, 0), 1)

    dst = 0
    if approx_contour is not None and len(approx_contour) == 4:
         approx_contour = np.float32(approx_contour)
         dst = four_point_transform(threshold, approx_contour)
    croppedImage = dst
    cleanedImage = final_image(croppedImage)

    return cleanedImage

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def final_image(rotated):
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
    sharpened = increase_brightness(sharpened, 30)
    return sharpened

def take_test():
    generate_practice(False)
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



