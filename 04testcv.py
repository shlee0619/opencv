import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager


import numpy as np
import cv2

#01testcv.py

path = './data/aveng.jpg'
img = cv2.imread(path)
cv2.imshow('first try', img)
cv2.waitKey() #필수기술

# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
cv2.imshow('color test', img2)
cv2.waitKey()


# img3 = cv2.imread(path, cv2.COLOR_BGR2GRAY)
# cv2.imshow('3 test', img3)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV test', hsv_img)
cv2.imshow('y grayscale test', hsv_img[ : , : , 0]) #grayscale
cv2.waitKey()
time.sleep(1)
cv2.imshow('y1 test', hsv_img[ : , : , 1]) #1
cv2.imshow('y2 test', hsv_img[ : , : , 2]) #2
cv2.waitKey()



print('11-1-금요일 opencv test 10 28')