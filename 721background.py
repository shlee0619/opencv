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
img_rgb= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('original 원본이미지')
plt.show() 

rectangle = (1,2,3,4)
mask = np.zeros()

#임시배열
bgmask = np.zeros()
fgmask = np.zeros()

cv2.grabCut(1,2,3,4,5,6,7)

a = 1
b = 2
mask = np.where(a|b)

#mask곱연산
img_cal = mask*mask
plt.imshow(img_rgb)
plt.title('11-05-화요일 배경제거 grabCut함수')
plt.show()


#배경제거후 픽셀 중간값을 계산해서 Canny()
med = np.median(img)
cv2.Canny() #GaussianBlur(), Canny()




# a, img_bin = cv2.threshold(img_rgb, 90, 155, cv2.THRESH_BINARY)
# print('cv2.threshold함수 첫번째 리턴값 ' , a)
# print('cv2.threshold함수 두번째 리턴값 ' , img_bin)
# plt.imshow(img_bin)
# plt.show() 


# adt_gaus = cv2.adaptiveThreshold(1,2,3,4,5,6)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
adt_gaus = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15,2)
#평균 adt_gaus = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15,2)

plt.imshow(adt_gaus, cmap='gray')
plt.show() 

#PDF문서 71페이지 cv2.adaptiveThreshold()함수 예제 기술
print()
print()



