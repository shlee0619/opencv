import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2

path = './data/a1.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('original 원본이미지')
plt.axis('off')
plt.show()

#블러는 각 픽셀=화소에 커널개수 곱해서 처리

# kernel = np.ones((10,10))/25.0
kernel = np.ones((10,10))/50.0
# kernel = np.ones((10,10))/10.0
# kernel = np.ones((10,10))/100.0
# kernel = np.ones((10,10)) #최소 red,yellow색만 추출 비추천
# 비권장 kernel = np.ones((10,10)) #blur(), threshold(), 가우시안블러

image_kernel = cv2.filter2D(img, -1, kernel)
plt.imshow(image_kernel, cmap='gray')
plt.show()
print()
print()

print(kernel)
print('합계 ', np.sum())