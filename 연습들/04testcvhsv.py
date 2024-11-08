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
cv2.imshow('1 test', img)
cv2.waitKey() #필수기술 

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# cv2.imshow('원본 hsv_img ', hsv_img)
cv2.imshow('hsv_img[:, :, 0]', hsv_img[:, :, 0]) 
cv2.imshow('hsv_img[:, :, 1]', hsv_img[:, :, 1]) 
cv2.imshow('hsv_img[:, :, 2]', hsv_img[:, :, 2]) 
cv2.waitKey() #필수기술 

# HSV  Hue(색상)   Saturation(채도)   Value(명도)
print('11-1-금요일 opencv test  11시 05분 ')





