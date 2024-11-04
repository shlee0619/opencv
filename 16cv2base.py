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

path = './data/~~~'
img = cv2.imread(path)
cv2.imshow('test ', img)  #plt.imshow() plt.show()
cv2.waitKey() #필수기술 






print('11-04-월요일  test')
print()
print()



