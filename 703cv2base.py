import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager


import numpy as np
import cv2
#------------------------------------------------------------------

path = './data/39.png'
img = cv2.imread(path)
print(img)
print()
print('img.shape ', img.shape)
print()
print('img.size ', img.size)
print('img.dtype ', img.dtype)
print('- '*70)
cv2.imshow('test', img)
cv2.waitKey()



print('11-04-월요일 test')
print()
print()