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
cv2.waitKey() 

#1번째 블러1, 커널1 수작업
# kernel1 = (2/16) * np.array([ [1,2,1], [2,4,2], [1,2,1] ] )
kernel1 = (1/16) * np.array([ [3,2,1], [7,4,2], [1,2,12] ] )
blur1 = cv2.filter2D(img, -1, kernel1)
cv2.imshow('filter2D', blur1)


#2번째 블러2, 커널2 getGaussianKernel(3)
# kernel2 = cv2.getGaussianKernel(3,0)
# blur2 = cv2.filter2D(img, -1, kernel2*kernel2.T)
# cv2.imshow('blur2', blur2)

#3번째 블러2, 커널2GaussianBlur(원본이미지, 3)
# blur3 = cv2.GaussianBlur(img, (3,3), 0)
# cv2.imshow('blur3', blur3)
cv2.waitKey() 







print()

