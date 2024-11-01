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
# https://opencv.org/


img = cv2.imread('./data/a1.png')
cv2.imshow('title 1', img)
cv2.waitKey()

yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('title 2', yuv_img)
cv2.waitKey()
print()
print('이미지출력 해야 합니다 ')


