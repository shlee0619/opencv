import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
import os
import numpy as np
import cv2  
# pip install opencv-python
#-----------------------------------------------------------------------------------
import face_recognition
# pip install cmake 먼저설치후 
# pip install face_recognition 설치 


#1  iu사진출력
# path = './data/36.jpg' 

path = './data/obama.jpg'

img = cv2.imread(path)
cv2.imshow('iu test', img)
cv2.waitKey() 

# path = './data/37.jpg'  #path = './data/37.jpg'  
path = './data/obama.jpg'
img = cv2.imread(path)
cv2.imshow(' test', img)
cv2.waitKey() 

# imga = face_recognition.load_image_file('./data/37.jpg')
imga = face_recognition.load_image_file('./data/biden.jpg')
imga = cv2.cvtColor(imga,cv2.COLOR_BGR2RGB)  

# imgTest = face_recognition.load_image_file('./data/36.jpg')  
imgTest = face_recognition.load_image_file('./data/obama.jpg')  
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) 
print( 'face_recognition  11111111111  2시 11분 ')

#얼굴의 위치 같게 만들기
faceLoc = face_recognition.face_locations(imga)[0]
encodeAA = face_recognition.face_encodings(imga)[0]   
cv2.rectangle(imga,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,255),2) #얼굴위치 확인하기 위해 사각형을 이미지에 그림
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]   
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,255),2)
print( 'face_recognition   2222222222 2시 15분 ')

results = face_recognition.compare_faces([encodeAA],encodeTest) #인코딩 이미지와 Test 이미지 간의 비교
faceDis = face_recognition.face_distance([encodeAA],encodeTest) #이미지 유사성 알기
print(results, faceDis) #두 개의 이미지가 서로 같으면 True, 다른 이미지일 경우 False를 출력
 

cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
 
cv2.imshow('imga', imga)  
cv2.imshow('iu', imgTest) 
cv2.waitKey(0)

print( 'iu이미지 비교 끝  ')