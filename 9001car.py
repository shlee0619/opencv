import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# pip install opencv-python
# https://opencv.org/


# 단독실습 1
path = './images/car/05.jpg'
img_origin = cv2.imread(path, cv2.IMREAD_COLOR) #imread_color
height, width, channel = img_origin.shape
cv2.imshow('car 61 1099 ', img_origin) 
cv2.waitKey(0)

time.sleep(1)
dx = cv2.Sobel(img_origin, -1, 1, 0)
dy = cv2.Sobel(img_origin, -1, 0, 1)

cv2.imshow(' dx Sobel  -1, 1, 0 ', dx) 
cv2.imshow(' dy Sobel  -1, 0, 1 ', dy) 
cv2.waitKey(0)
print()



gray = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HSV) #블루색 더강함 
# plt.figure(figsize=(10, 6)) 
plt.title('cv2.COLOR_RGB2HSV')  
plt.imshow(gray,cmap='gray')
plt.show()
time.sleep(1)


#Thresholding 글자번호를 기준으로  윤곽선표시 
print('- ' * 60)
gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
img_thresh = cv2.adaptiveThreshold(
    gray,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
plt.title('cv2.adaptiveThreshold')
plt.imshow(img_thresh, cmap='gray')
plt.show()
time.sleep(1)




# 가우시안블러 - 그림의 노이즈 제거 
# Contours를 찾으려면 검은색 배경에 흰색 바탕이어야 함
# Contours란 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
#4번째 GauusianBlur적용해서 Thresholding 글자번호추출
hsv = cv2.cvtColor(img_origin, cv2.COLOR_BGR2HSV)
gray = hsv[:, :, 2]
img_blur = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
img_blur_thresh = cv2.adaptiveThreshold(
    img_blur,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

plt.title('cv2.GaussianBlur')
plt.imshow(img_blur_thresh, cmap='gray')
plt.show()
# plt.axes('off')
print()

##########################################################################################################


'''
cv2.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None) 
 src: 입력 영상
 ddepth: 출력 영상 데이터 타입. -1이면 입력 영상과 같은 데이터 타입을 사용.
 dx: x 방향 미분 차수. 1차미분 2차미분 
 dy: y 방향 미분 차수.
 dst: 출력 영상(행렬)
 ksize: 커널 크기. 기본값은 3.
 scale 연산 결과에 추가적으로 곱할 값. 기본값은 1.
 delta: 연산 결과에 추가적으로 더할 값. 기본값은 0.
 borderType: 가장자리 픽셀 확장 방식. 기본값은 cv2.BORDER_DEFAULT.
'''

