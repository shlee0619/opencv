import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager


import numpy as np
import cv2
#------------------------------------------------------------------

path = './data/39.png'
img = cv2.imread(path, cv2.IMREAD_COLOR)
print(img)
print('img.size ', img.size)
print('img.dtype ', img.dtype)
print()
img_100 = cv2.resize(img, (100,100))
img_50 = cv2.resize(img, (50,50))


fit, ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(img)
ax[0].set_title('original img')
ax[1].imshow(img)
ax[1].set_title('100size')
ax[2].imshow(img)
ax[2].set_title('100size img')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()





print('11-04-월요일 test')
print()
print()