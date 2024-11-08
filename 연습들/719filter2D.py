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
img = cv2.imread(path) # cv2.IMREAD_GRAYSCALE)
print(img)
img_rgb= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
print('img_rgb.shape ' ,img_rgb.shape) 
plt.title('a1.png이미지')
plt.axis('off') 
plt.show() 


#filter2D()
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]) 
image_kernel = cv2.filter2D(img, 0, kernel)
plt.imshow(image_kernel, cmap='gray')
plt.show()







print()
print()



