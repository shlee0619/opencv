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

img3 = cv2.imread('./data/corgi.jpg')
cv2.imshow('corgi.jpg', img3)
cv2.waitKey()

#pDF문서 167페이지  cv2.threshold( )
#res, img3 = cv2.threshold(img3, 127, 255, cv2.THRESH_BINARY_INV)
res, img3 = cv2.threshold(img3, 200, 200, cv2.THRESH_TRUNC)

#pDF문서 157페이지  cv2.morphologyEx()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
opening = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img3, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img3, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img3, cv2.MORPH_BLACKHAT, kernel)

titles = ['original', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']
images = [img3, opening, closing, gradient, tophat, blackhat]

plt.figure(figsize=(8, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    
plt.tight_layout()
plt.show()

'''
Morphological Transformation 
 1. Erosion  : (침식) 원본 이미지의 각 픽셀에 structuring element(구조화 요소)를 적용해 하나라도 0이 있으면 대상 픽셀을 제거하는 방법이다.
 2. Dilation : (팽창) 팽창은 원본 이미지의 각 픽셀에 structuring element(구조화 요소)를 적용해 하나라도 1이 있으면 대상 픽셀을 1로 만드는 방법이다.
 3. Opening : 이미지에 Erosion(침식) 적용 후 Dilation(팽창) 적용하는 것으로 영역이 점점 둥글게 된다. 따라서 점 잡음 이나 작은 물체, 돌기 등을 제거하는데 적합하다.
 4. Closing : 이미지에 Dilation(팽창) 적용 후 Erosion(침식) 적용하는 것으로 영역과 영역이 서로 붙기 때문에 이미지의 전체적인 윤곽을 파악하기에 적합하다.
'''

print()