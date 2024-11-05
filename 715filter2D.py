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
#715filter2D.py문서작성
path = './data/a1.png' 
img = cv2.imread(path)
img_rgb= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('original 원본이미지')
plt.axis('off') 
plt.show() 

# blur(), cv2.adaptiveThreshold(), 가우시안블러()
kernel = np.ones((10,10)) /50.0   #적절
image_kernel = cv2.filter2D(img, -1, kernel)
plt.imshow(image_kernel, cmap='gray')
plt.show()




print()
print()



