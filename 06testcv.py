import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2  
#-----------------------------------------------------------------------------------

path = './data/a1.png'
img = cv2.imread(path) 
# img = cv2.imread(path , cv2.IMREAD_GRAYSCALE) 
# img = cv2.imread(path , cv2.IMREAD_COLOR) 
cv2.imshow('test 1', img) #컬러그대로 적용
cv2.waitKey()


# img = cv2.imread(path , cv2.IMREAD_COLOR) 
img = cv2.imread(path)

b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])

# 컬러가 다르게 출력되어서 다시 분리,머지시킴 plt.imshow(img) #RGB대신 BGR표현
plt.imshow(img2) #RGB대신 BGR표현
plt.xticks([])
plt.yticks([])
plt.title('cv2.IMREAD_COLOR')
plt.show()

# RGB =  Red, Green, Blue
# HSV =  Hue(색상)   Saturation(채도)   Value(명도)







print('11-1-금요일 opencv test ')





