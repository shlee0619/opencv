import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2  
#-----------------------------------------------------------------------------------

def visualize_image(title, image, cmap_type='gray'):
    """이미지 시각화 함수"""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 이미지 경로 설정
path = './data/a1.png' 
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"이미지를 '{path}'에서 찾을 수 없습니다.")

# BGR 이미지를 RGB로 변환하여 시각화
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
visualize_image('Original 원본이미지', img_rgb)

# GrabCut을 위한 사각형 정의 (x, y, width, height)
rectangle = (0, 56, 256, 150)  # 이미지에 맞게 조정 필요
mask = np.zeros(img.shape[:2], np.uint8)

# GrabCut을 위한 임시 배열
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 실행
cv2.grabCut(img, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 마스크 시각화
mask_visual = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8') * 255
visualize_image('GrabCut 마스크', mask_visual)

# 마스크 적용하여 배경 제거
mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
img_rgb_ob = img_rgb * mask2[:, :, np.newaxis]

# 마스킹된 이미지 시각화
visualize_image('마스킹된 이미지', img_rgb_ob)

print('이미지 마스킹 완료')
