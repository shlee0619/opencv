import cv2
import numpy as np


def minmax(pixel):
    if pixel > 255:
        pixel = 255
    if pixel < 0:
        pixel = 0
    return pixel

def dithering_gray(inMat, samplingF):
    h = inMat.shape[0]
    w = inMat.shape[1]

    for y in range(0, h - 1):
        for x in range(1, w - 1):
            old_p = inMat[y, x]
            new_p = np.round(samplingF * old_p / 255.0) * (255 / samplingF)
            inMat[y, x] = new_p

            quant_error_p = old_p - new_p

            inMat[y, x + 1] = minmax(inMat[y, x + 1] + quant_error_p * 7 / 16.0)
            inMat[y + 1, x - 1] = minmax(inMat[y + 1, x - 1] + quant_error_p * 3 / 16.0)
            inMat[y + 1, x] = minmax(inMat[y + 1, x] + quant_error_p * 5 / 16.0)
            inMat[y + 1, x + 1] = minmax(inMat[y + 1, x + 1] + quant_error_p * 1 / 16.0)
    return inMat

# 이미지 가져오기
lena = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# 눈 이미지 추출하기 위한 xml 파일들 가져오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 얼굴 
faces = face_cascade.detectMultiScale(lena, 1.3, 5)

for (x, y, w, h) in faces:
    roi_face = lena[y:y+h, x:x+w]  # 얼굴 영역

    # face roi 로 눈 확인
    eyes = eye_cascade.detectMultiScale(roi_face)
    for (ex, ey, ew, eh) in eyes:
        roi_eye = roi_face[ey:ey+eh, ex:ex+ew]  # 눈 영역
        lena[y+ey:y+ey+eh, x+ex:x+ex+ew] = dithering_gray(roi_eye.copy(), 1)  # 눈쪽에만 dithering

# Save and display the result
cv2.imwrite('./data/lena_eye_dithering.jpg', lena)
cv2.imshow('Lena with Eye Dithering', lena)
cv2.waitKey(0)
cv2.destroyAllWindows()
