import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2  
#-----------------------------------------------------------------------------------

path = './data/a1.png' 
img = cv2.imread(path)
img_rgb= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('717gaussianblur 원본이미지')
plt.axis('off') 
plt.show() 

#GaussianBlur( ) 특이하게 첫글자 대문자시작 
image_kernel = cv2.GaussianBlur(img, (5,5), 0)
plt.imshow(image_kernel, cmap='gray')
plt.show()




print()
print()



