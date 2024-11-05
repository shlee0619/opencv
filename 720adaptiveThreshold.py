import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
import numpy as np
import cv2  

# 한글 폰트 설정
font_path = 'c:/windows/Fonts/malgun.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

# 이미지 경로 설정 (사용자 입력 가능)
path = './data/a1.png' 
img = cv2.imread(path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


a, img_bin = cv2.threshold(img_rgb, 90, 155, cv2.THRESH_BINARY)
print(a)
print(img_bin)
plt.imshow(img_bin)
plt.show()


# adt_gaus= cv2.GaussianBlur(1,2,3,4,5,6)
adt_gaus= cv2.GaussianBlur(1,2,3,4,5,6)