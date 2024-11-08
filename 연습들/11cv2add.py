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
img1 = cv2.imread('./data/wing_wall.jpg')
img2 = cv2.imread('./data/yate.jpg')
img3 = img1 + img2  # 더하기 연산
img4 = cv2.add(img1, img2) # OpenCV 함수

imgs = {
    'img1':img1, 'img2':img2, 
    'img1+img2':img3, 'cv.add(img1, img2)':img4}

for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([])
    plt.yticks([])

plt.show()
print()

print('11cv2add.py문서 end')