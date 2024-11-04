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

path = './data/39.png'
img = cv2.imread(path)
print(img)
print()
print(img.shape)
print()
print('img.size ' , img.size)
print('img.dtype ' , img.dtype)  
print('- ' * 70 )
cv2.imshow('test ', img)  #plt.imshow() plt.show()
cv2.waitKey() #필수기술


# 이미지 정보 height, width, channel = img_origin.shape
# 700시리즈  컬러변경, 사이즈변경, 자르기, 기타 YUV, HSV, 컬러변경 cvtColor
# 700시리즈  blur처리하면서 kernel, filter2D, GaussianBlur
# 700시리즈  이미지 2진화 cv2.adaptiveThreshold(1,2,3,4,5,6)
# 700시리즈  마스크생성, Canny()
print()
print()



