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

rectangle = (0,56,256,150)
mask = np.zeros(img_rgb.shape[:2], np.uint8)

#임시배열
bgmask = np.zeros((1,65), np.float64)
fgmask = np.zeros((1,65), np.float64)

#cv2.grabCut(1src, 2마스킹, 3사각형, 4백, 5프런트, 6반복횟수, 7사각형초기화)
cv2.grabCut(img_rgb, mask, rectangle, 
            bgmask, fgmask, 5, cv2.GC_INIT_WITH_RECT)
# cv2.grabCut(1,2,3,4,5,6,7)

mask2=np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_rgb_ob = img_rgb * mask2[:,:, np.newaxis]
plt.imshow(img_rgb_ob)
plt.title('14:40분 마스킹 이미지')
plt.show()
print('np & cv2')


img_gray = cv2.imread('./data/a1.png', cv2.IMREAD_GRAYSCALE)
img_md = np.median(img_gray)
print('이미지 미디언 ', img_md)
plt.imshow(img_gray)
plt.title('11-05-화요일 gray test')
plt.show()



lowera = int(max(0,(1.0-0.33)*img_md))
upperb = int(min(255,(1.0+0.33)*img_md))
image_canny  = cv2.Canny(img_gray, lowera, upperb)
plt.imshow(image_canny, cmap = 'gray')
plt.title('11-05-화요일 canny test')
plt.show()



'''

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



'''