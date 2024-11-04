import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
font_name = font_manager.FontProperties(fname = 'c:/windows/Fonts/malgun.ttf')
rc('font', family=font_name)

import numpy as np
import cv2
#------------------------------------------------------------------

path = './data/~~~'
img = cv2.imread(path)
cv2.imshow('test', img)
cv2.waitKey()

# 700시리즈 컬러변경, 사이즈변경, 자르기, 기타 YUV, HSV
# 700시리즈 blur처리하면서 kernel, filter2D, GaussianBlur
# 700시리즈 이미지 2진화 cv2.adaptiveThreshold(1,2,3,4,5)

print('11-04-월요일 test')
print()
print()