import cv2
import sys
import os

# 윈도우 타이틀과 이미지 불러오기
title = 'mouse event'
image_path = os.path.join('data', 'blank_500.jpg')  # 상대 경로 사용
img = cv2.imread(image_path)
if img is None:
    print(f"이미지를 불러올 수 없습니다: {image_path}")
    sys.exit()

cv2.imshow(title, img)

# 색상 사전 정의
colors = {
    'black': (0, 0, 0), 'red': (0, 0, 255),
    'blue': (255, 0, 0), 'green': (0, 255, 0)
}

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    color = colors['black']
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['green']
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['blue']
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            color = colors['red']

        cv2.circle(img, (x, y), 50, color, -1)
        cv2.imshow(title, img)

cv2.setMouseCallback(title, onMouse)

# ESC 키를 누르면 종료
if cv2.waitKey(0) & 0xFF == 27:
    print("프로그램 종료")
    cv2.destroyAllWindows()
    sys.exit()
