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

import cv2
import numpy as np

# 이미지 로드
path = './data/a1.png'
img = cv2.imread(path)

# 원본 이미지 표시
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# 1. 더 큰 커널을 사용한 필터링
kernel_size = 5
kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=0)
kernel = kernel * kernel.T  # 2D 가우시안 커널 생성
blur_large_kernel = cv2.filter2D(img, -1, kernel)

# 2. cv2.GaussianBlur 함수 사용
blur_gaussian = cv2.GaussianBlur(img, (11, 11), 0)

# 3. 필터를 여러 번 적용
blur_multiple = img.copy()
for _ in range(5):
    blur_multiple = cv2.filter2D(blur_multiple, -1, kernel)

# 결과를 가로로 이어붙이기
combined = np.hstack((img, blur_large_kernel, blur_gaussian, blur_multiple))

# 결과 이미지 표시
cv2.imshow('Original | Large Kernel | GaussianBlur | Multiple Filters', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()







