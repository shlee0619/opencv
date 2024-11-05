import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
import numpy as np
import cv2  

# 한글 폰트 설정
font_path = 'c:/windows/Fonts/malgun.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

# 이미지 경로 설정 (사용자 입력 가능)
path = './data/a1.png' 
# path = input("이미지 파일 경로를 입력하세요: ")

# 이미지 읽기 (컬러)
img = cv2.imread(path)

# 이미지 로드 확인
if img is None:
    print(f"이미지를 경로 '{path}'에서 찾을 수 없습니다.")
    exit()

# BGR을 RGB로 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 원본 이미지 출력
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title('Original 원본 이미지', fontsize=16)
plt.axis('off')
plt.show()

# 그레이스케일로 변환
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이진화 (Otsu의 방법 사용)
threshold_value, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Otsu 임계값 >>>', threshold_value)

# 이진화된 이미지 출력
plt.figure(figsize=(8, 6))
plt.imshow(img_bin, cmap='gray')
plt.title('이진화된 이미지 (Otsu)', fontsize=16)
plt.axis('off')
plt.show()

# 이진화된 이미지 저장
output_path = './data/a1_bin.png'
cv2.imwrite(output_path, img_bin)
print(f"이진화된 이미지를 '{output_path}'로 저장했습니다.")
