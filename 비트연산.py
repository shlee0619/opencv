import cv2
import numpy as np

# 비트 연산을 적용하기 전에 모든 차원과 타입을 명시적으로 확인
# 이전 작업에서 변경된 내용을 피하기 위해 이미지를 다시 로드

img1 = cv2.imread('./data/logo.png')  # 로고 이미지 (Apple 로고)
img2 = cv2.imread('./data/lena.png')  # 배경 이미지 (Lena)

img1_resized = cv2.resize(img1, (225, 225))  # 로고 이미지를 (225, 225) 크기로 조정

# 조정 후 새로운 크기 가져오기
rows, cols, channels = img1_resized.shape

# 배경 이미지 (img2)에서 img1_resized와 동일한 크기의 관심 영역 (ROI) 정의
roi = img2[0:rows, 0:cols]

# img1_resized를 그레이스케일로 변환하여 마스크 및 반전 마스크 생성
img1gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img1gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# mask와 mask_inv가 roi의 크기 및 타입과 일치하도록 설정
mask = mask.astype(np.uint8)
mask_inv = mask_inv.astype(np.uint8)

# mask_inv를 3채널로 변환하여 roi와 일치시키기
mask_inv_rgb = cv2.merge([mask_inv, mask_inv, mask_inv])

# 마스크를 적용하여 전경 이미지와 배경 이미지 생성
img1_fg = cv2.bitwise_and(img1_resized, img1_resized, mask=mask)
img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# 전경과 배경 결합
dst = cv2.add(img1_fg, img2_bg)

# 결합된 이미지를 원본 이미지 (img2)의 해당 위치에 배치
img2[0:rows, 0:cols] = dst

# 결과 출력
cv2.imshow('Result', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
