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

#06testcv.py
path = './data/aveng.jpg'
# img = cv2.imread(path)
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(path, cv2.IMREAD_COLOR)

img = cv2.imread(path, cv2.IMREAD_COLOR)
plt.imshow(img )
plt.show()
cv2.waitKey()


# cv2.filter2D()




print('11-1 금요일 opencv test 11 39')
print()


