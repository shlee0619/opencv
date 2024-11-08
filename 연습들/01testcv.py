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

#01testcv.py
path = './data/a1.png'
img = cv2.imread(path)

print(img) # 넘파이배열로반환
print()
print('img.shape ' ,img.shape) #튜플형태 3개값 채널 
print('img.size ' , img.size)
print('img.dtype ' , img.dtype)  
print('- ' * 70 )
cv2.imshow('test 1', img)
cv2.waitKey() #필수기술 

# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# img2 = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
cv2.imshow('test', img2 )
cv2.waitKey() #필수기술 


print('11-04-월요일 opencv test')





