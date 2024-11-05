import numpy as np
import cv2 
import sys
import time
import matplotlib.pyplot as plt

# pip install opencv-python
# https://opencv.org/


# path ='./data/scuba.mp4'
# video = cv2.VideoCapture(path)
# check, frame = video.read()
# cv2.imshow('test', frame)
# cv2.waitKey()
    
# video.release() #동영상을 종료하면 사용했던 메모리 자원을 반환
#cv2.destroyAllWindows()   #모든창 닫기  


time.sleep(1)
path ='./data/scuba.mp4'
video = cv2.VideoCapture(path)

num=0
while video.isOpened():
    check, frame = video.read()
    if not check:
        print('동영상 영상 끝났습니다')
        break
    
    cv2.imshow('test', frame)
    mypath = './my/myscuba_'+ str(num)+'.jpg'
    cv2.imwrite(mypath, frame)  #myscuba_0.jpg ~ myscuba_218.jpg생성됨
    if cv2.waitKey(25) == ord('q'):
        print('영상을 강제종료합니다')
        break
    num = num + 1

video.release() #동영상을 종료하면 사용했던 메모리 자원을 반환
cv2.destroyAllWindows()   #모든창 닫기 
print('10-31-목요일 스쿠버 동영상 end ~~~')
