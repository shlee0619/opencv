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
 
img = np.full( (250,250,3), 255, dtype=np.uint8)
cv2.circle(img, (120,120), 100, (255,255,0), 15) #bgr

cv2.imshow('circle' , img)
cv2.waitKey()
cv2.destroyAllWindows() 
print('circle testing ')