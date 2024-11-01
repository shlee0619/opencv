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
path = './data/aveng.jpg'
img = cv2.imread(path)
cv2.imshow('1 test', img)
cv2.waitKey() 

# cv2.cvtColor () 실습확인
# 필터적용 cv2.filter2D(1이미지, 2)
kerne13 = np.ones((3,3), np.float32) / 9
kerne15 = np.ones((5,5), np.float32) / 25

filter3 = cv2.filter2D(img, -1, kerne13)
filter5 = cv2.filter2D(img, -1, kerne15)
cv2.imshow('cv2.filter2D(img, -1, kerne13)', filter3)
cv2.imshow('cv2.filter2D(img, -1, kerne15)', filter5)
cv2.waitKey()


# cv2.filter2D()




print('11-1 금요일 opencv test 11 14')
print()


