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


path = './data/36.jpg' 
img = cv2.imread(path)
print('img정보 ' , img)

mask = np.zeros( img.shape[:2] ,dtype=np.uint8) 
cv2.circle(mask, (170,150 ), 150, (255,0,0), -1)
masked = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('iu test', img)
cv2.imshow('mask', mask)
cv2.imshow('masked', masked)
cv2.waitKey() 








print()

