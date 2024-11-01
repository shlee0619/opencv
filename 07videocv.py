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

#06testcv.py
path = './data/Puppies-FHD.mp4'







print('11-1 금요일 Puppies-FHD.mp4 동영상')

