import time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

warnings.filterwarnings('ignore')

# 한글 폰트 설정



# 이미지 경로
path = './data/39.png'

# 이미지 불러오기
img = cv2.imread(path)
if img is None:
    print(f"이미지를 불러올 수 없습니다: {path}")
else:
    cv2.imshow('Original Image', img)
    cv2.waitKey()

# 700 시리즈 예제: 컬러 변경, 사이즈 변경, 자르기, YUV, HSV 변환
def process_image(img):
    # 이미지 사이즈 변경 (반으로 축소)
    resized_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # 이미지 컬러 변경 (BGR to GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HSV와 YUV 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 이미지 자르기 (가운데 부분만)
    cropped_img = img[50:200, 50:200]

    # 각 처리된 이미지 보여주기
    cv2.imshow('Resized Image', resized_img)
    cv2.imshow('Gray Image', gray_img)
    cv2.imshow('HSV Image', hsv_img)
    cv2.imshow('YUV Image', yuv_img)
    cv2.imshow('Cropped Image', cropped_img)

    cv2.waitKey()

# 700 시리즈 예제: Blur 처리
def apply_blur(img):
    # Blur 처리 (kernel 사이즈 5x5)
    kernel = np.ones((5, 5), np.float32) / 25
    blur_img = cv2.filter2D(img, -1, kernel)

    # GaussianBlur 적용
    gaussian_blur_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 블러 처리된 이미지 보여주기
    cv2.imshow('Blur Image', blur_img)
    cv2.imshow('Gaussian Blur Image', gaussian_blur_img)

    cv2.waitKey()

# 700 시리즈 예제: 이미지 2진화
def binary_threshold(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold 적용
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    binary_gaussian_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 이진화된 이미지 보여주기
    cv2.imshow('Binary Threshold (Mean)', binary_img)
    cv2.imshow('Binary Threshold (Gaussian)', binary_gaussian_img)

    cv2.waitKey()

# 각 처리 함수 실행
if img is not None:
    process_image(img)
    apply_blur(img)
    binary_threshold(img)

# 프로그램 종료
cv2.destroyAllWindows()

print('11-04-월요일 test')
print()
print()
