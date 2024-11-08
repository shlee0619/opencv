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
# path = './data/39.png' 

img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(path , cv2.IMREAD_COLOR)
# img = cv2.imread(path)

img_crop = img[ : , 50 : 200]
# plt.imshow(img) 원본그대로 출력 
plt.imshow(img_crop)
plt.xticks([])
plt.yticks([])
plt.show()


print()
print()



