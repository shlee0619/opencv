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


path = './data/a1.png'
img = cv2.imread(path)

h, w = img.shape[0:2] 
print('높이h =' , h )
print('길이w =' , w ) 
print('img.shape ' ,img.shape) #높이,가로,채널
cv2.imshow('test', img)
cv2.waitKey() 


print('- ' * 50) 
path ='./data/scuba.mp4'
cap = cv2.VideoCapture(path)

h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('높이h =' , h )
print('길이w =' , w ) 
print('frame수 =' , cnt )  #scuba프레임 0~281쪼개져서 총 219

ret, frame = cap.read() 
print('ret 결과 ' , ret)

# while반복문  if ret==True:
#     cv2.imshow('water' , frame)
#     cv2.waitKey()

# cv2.imshow('test', img)
# cv2.waitKey() 




