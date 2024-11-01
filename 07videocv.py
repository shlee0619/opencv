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




cap = cv2.VideoCapture(path)

dummy, pupp = cap.read()
print()
print('dummy 결과 불', dummy)
cv2.imshow('puppies dog', pupp)
cv2.waitKey()


# if cap.isOpened():
#     delay = int(1000/cap.get(cv2.CAP_PROP_FPS))
#     while True:
#         ret, img = cap.read()
#         if ret:
#             cv2.imshow("Movie", img)
#             if cv2.waitKey(delay) & 0xFF == 27:
#                 print("ESC Key pressed")
#                 break
#         else:
#             print(ret, img)
#             break
# else:
#     print("File not opened")





print('11-1 금요일 Puppies-FHD.mp4 동영상')

