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
path = './data/Puppies-FHD.mp4'
cap = cv2.VideoCapture(path)
# pupp = cap.imread() 에러
# pupp = cap.imread() 에러
ret, pupp = cap.read() 
print()
print('ret 결과 ' , ret)
if ret:
    pass
    cv2.imshow('puppies dog' , pupp)
    cv2.waitKey()


print('11-1-금요일 Puppies-FHD.mp4동영상')


# path = './data/a1.png'
# img = cv2.imread(path)
# cv2.imshow('test 1', img)
# cv2.waitKey() #필수기술 




