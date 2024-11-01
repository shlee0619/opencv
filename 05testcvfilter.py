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

#05testcvfilter.py
path = './data/a1.png'
img = cv2.imread(path)
cv2.imshow('1 test', img)

kernel3 = np.ones((3,3), np.float32) # / 9
kernel5 = np.ones((5,5), np.float32) # / 25

filter3 = cv2.filter2D(img, -1, kernel3)
filter5 = cv2.filter2D(img, -1, kernel5)
cv2.imshow('cv2.filter2D(img, -1, kernel3)',filter3)
cv2.imshow('cv2.filter2D(img, -1, kernel5)',filter5)
cv2.waitKey() 

#filter효과가 약해서 블러링 가우시안블러링 적용 

print('11-1-금요일 opencv test   11 14')
print()





