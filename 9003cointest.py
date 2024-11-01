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
# https://opencv.org/


#  scikit-learn버젼 
#  아님  pip install  sklearn 
#  정답  pip install  scikit-learn  
#  확인  pip list



## 1번째 testing ... 1
coin_no = 84 # 01  29  12   84
image = cv2.imread('./coin/%02d.png' %coin_no, cv2.IMREAD_COLOR)
print(image)
print()
height, width, channel = image.shape 
print('image.shape 정보', image.shape)
print('width=',width, 'height=', height,  'channel =', channel )
cv2.imshow('84 coin test ', image)
cv2.waitKey()
time.sleep(1)
print()