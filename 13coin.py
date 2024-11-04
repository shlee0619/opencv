import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2   

# img = cv2.imread('./data/a1.png')
# cv2.imshow(' 1 title', img)
# cv2.waitKey()

coin_no =  './coin/84.png'
img = cv2.imread(coin_no, cv2.IMREAD_COLOR) 
height, width, channel = img.shape 
print('11-04-월요일 img.shape 정보', img.shape)
print('11-04-월요일 width=',width, 'height=', height,  'channel =', channel )
cv2.imshow('11-04 test ', img)
cv2.waitKey()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          
blur = cv2.GaussianBlur(gray, (3,3), 0)    
circle = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 30, None, 200)

if circle is not None:
    circle = np.uint16(np.around(circle))
    
    for k in circle[0, :]:
        cv2.circle(img, (k[0], k[1]), k[2], (0,255,0), 2) 
        cv2.circle(img, (k[0], k[1]), 2, (0,0,255), 5) 


cv2.imshow('11-04 test ', img)
cv2.waitKey()
cv2.destroyAllWindows()
print()

