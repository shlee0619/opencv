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

yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
cv2.imshow('color test', img2)
cv2.waitKey()



print('11-1-금요일 opencv test 10 13')