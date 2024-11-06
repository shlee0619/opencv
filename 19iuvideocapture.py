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


#1  iu사진출력
path = './data/36.jpg' 
img = cv2.imread(path)
print('img정보 ' , img)
cv2.imshow('iu test', img)
cv2.waitKey() 


# 동영상이라서 시간이 오래걸림  사각형으로 사람모델가중치
body_cascade = cv2.CascadeClassifier('./data/haarcascade_fullbody.xml')
cap = cv2.VideoCapture('./data/mov.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('face Frame ', frame)
    if cv2.waitKey(1) | 0xFF == ord('q') :  #종료
        break

cap.release()
cv2.destroyAllWindows()

