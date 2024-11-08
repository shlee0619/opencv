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

# path = './data/a1.png' 
path = './data/39.png' 
# path = './data/hydran.png'  

# img = cv2.imread(path)  #기본 원색컬러 그대로 유지
# img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
img = cv2.imread(path , cv2.IMREAD_COLOR)
print('img.size ' , img.size)
print('img.dtype ' , img.dtype)  

plt.imshow(img) 
plt.axis('off') 
plt.show() 

# RGB 변환 
img_rgb= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off') 
plt.show() 

# # RGB 변환 다르게 접근 참고
# img = cv2.imread(path)
# b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])


# cv2.imshow('test ', img)  #원본색그대로유지
# cv2.waitKey()             #잠시대기=키값입력 대기중 

# 이미지 정보 height, width, channel = img_origin.shape
# 700시리즈  컬러변경, 사이즈변경, 자르기, 기타 YUV, HSV, 컬러변경 cvtColor
# 700시리즈  blur처리하면서 kernel, filter2D, GaussianBlur
# 700시리즈  이미지 2진화 cv2.adaptiveThreshold(1,2,3,4,5,6)
# 700시리즈  마스크생성, Canny()
print()
print()



