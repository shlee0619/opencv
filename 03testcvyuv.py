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

#03testcv.py
path = './data/a1.png'
img = cv2.imread(path)
cv2.imshow('1 test', img)
cv2.waitKey() #필수기술 

yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# cv2.imshow('yuv  test', yuv_img)
cv2.imshow('yuv_img[:, :, 0]', yuv_img[:, :, 0]) 
cv2.imshow('yuv_img[:, :, 1]', yuv_img[:, :, 1]) 
cv2.imshow('yuv_img[:, :, 2]', yuv_img[:, :, 2]) 
cv2.waitKey() #필수기술 


print('11-1-금요일 opencv test  10 42 ')





