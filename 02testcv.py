import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2  
# pip install opencv-python
#-----------------------------------------------------------------------------------


path = './data/a1.png'
img = cv2.imread(path)
cv2.imshow('111 test', img)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('222 test', img2 )

img3 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('333 test', img3)
cv2.waitKey()  


print('11-4-월요일 ')

# RGB란 가장 일반적인 Color Space로 Red, Green, Blue
# Y밝기 UV 
# HSV  Hue(색상)   Saturation(채도)   Value(명도)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# img2 = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)





