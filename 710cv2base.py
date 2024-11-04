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
plt.axis('off')
plt.show()

img_blur = cv2.blur(img, (10,10))
plt.imshow(img_blur, cmap='gray')

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)



# 블러blur는 
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()