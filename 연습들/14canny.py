import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2   


img = cv2.imread('./data/corgi.jpg', cv2.IMREAD_COLOR)
b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])
# plt.imshow(img2) 
cv2.imshow('original', img)
cv2.waitKey()


img = cv2.imread('./data/corgi.jpg',cv2.IMREAD_GRAYSCALE) 
canny1 = cv2.Canny(img, 50, 200)
canny2 = cv2.Canny(img, 100, 200)
canny3 = cv2.Canny(img, 170, 200)

titles = ['original ', 'canny1 50/200', 'canny2 100/200', 'canny3 170/200']
images = [img, canny1, canny2, canny3]


plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    
plt.tight_layout()
plt.show()
print()
print('- ' * 50)



def onChange(a):
    print('onChange(a)')
    pass


def edge_detection():
    img = cv2.imread('./data/corgi.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('edge detection')
    
    cv2.createTrackbar('low', 'edge detection', 0, 255, onChange)
    cv2.createTrackbar('high', 'edge detection', 0, 255, onChange)
    cv2.imshow('edge detection', img)
    
    while True:
        k = cv2.waitKey(0) #  & 0xFF
        print('k =' , k)
        if k == 27:
            break 
            
        low = cv2.getTrackbarPos('low', 'edge detection')
        high = cv2.getTrackbarPos('high', 'edge detection')
        
        if low > high:
            print("Low수치는 higt보다 적은값이어야 합니다")
        
        elif ((low == 0) and (high == 0)):
            cv2.imshow('edge detection', img)
        
        else:
            canny = cv2.Canny(img, low, high)
            cv2.imshow('edge detection', canny)
    


# 함수호출
edge_detection()
print()
