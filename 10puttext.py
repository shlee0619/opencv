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
 
#pdf문서 47페이지 참고  
img = np.full( (450,350,3), 255, dtype=np.uint8)

cv2.putText(img, 'summer', (20,200), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))
cv2.imshow('text' , img)
cv2.waitKey()
print('putText testing ')