import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 이미지 번호 설정
car_no = 5  # 처리할 이미지 번호 (05, 13, 15, 20 등)

# 전처리 함수 정의
def preprocessing(car_no):
    print('\n전처리 함수 호출: preprocessing(car_no)')
    image_path = f'./images/car/{car_no:02d}.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"이미지를 '{image_path}'에서 찾을 수 없습니다.")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, th_img = cv2.threshold(gray, 130, 255, flag)
    kernel = np.ones((5, 13), np.uint8)  # 마스크용 커널 (폭이 긴 직사각형)
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return image, morph

# 번호판 크기 조건 검사 함수 정의
def car_size(size):
    w, h = size
    if w == 0 or h == 0:
        return False
    
    data = h / w if h > w else w / h
    chk1 = 3000 < (h * w) < 12000  # 번호판 넓이 조건
    chk2 = 2.0 < data < 6.5          # 너비와 높이의 비율 조건
    return (chk1 and chk2)

# OpenCV 버전에 따라 findContours 반환값 처리
def get_contours(image):
    contours_info = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[-2]  # OpenCV 4.x 이상
    hierarchy = contours_info[-1]
    return contours, hierarchy

# 번호판 후보 찾기 함수 정의
def car_find(morph):
    print('\n번호판 후보 찾기 함수 호출: car_find(morph)')
    contours, _ = get_contours(morph)
    rects = [cv2.minAreaRect(ct) for ct in contours]
    csa = [(tuple(map(int, center)), tuple(map(int, size)), angle) for center, size, angle in rects if car_size(size)]
    return csa 

# 번호판 문자 추출 함수 정의
def car_charnumber(image, center):
    print('\n번호판 문자 추출 함수 호출: car_charnumber(image, center)')
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)
    dif1, dif2 = (25, 25, 25), (25, 25, 25)
    flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    # 중앙 근처에 Flood Fill 수행
    pts = np.random.randint(-15, 15, (20, 2)) + center
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            _, _, fill, _ = cv2.floodFill(image.copy(), fill, (x, y), 255, dif1, dif2, flags)

    # 이진화
    _, binary = cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)
    return binary

# 번호판 위치 추출 함수 정의
def car_position(image, rect):
    print('\n번호판 위치 추출 함수 호출: car_position(image, rect)')
    center, (w, h), angle = rect 

    if w < h:
        w, h = h, w
        angle -= 90

    size = (image.shape[1], image.shape[0])
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)
    crop_img = cv2.getRectSubPix(rot_img, (int(w), int(h)), center)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop_img, (144, 28))

# 전처리 수행
image, morph = preprocessing(car_no)
if image is None:
    exit()

# 번호판 후보 찾기
mycar = car_find(morph)
print(f"번호판 후보 수: {len(mycar)}")

# 번호판 문자 추출
fills = [car_charnumber(image, center) for center, size, angle in mycar]

# 문자 Contour 찾기 및 번호판 추출
new_cd = []
for fill in fills:
    contours, _ = get_contours(fill)
    rects = [cv2.minAreaRect(ct) for ct in contours]
    filtered = [rect for rect in rects if car_size(rect[1])]
    if filtered:
        new_cd.append(filtered[0])

############의견###############
# 작동은 오류없이 이루어지는데, imgs 리스트에 저장되는게 안되는 것 같다. 


# 번호판 위치 추출 및 저장
imgs = [car_position(image, k) for k in new_cd]
print('자동차 번호만 추출 성공')

# 추출된 번호판 이미지 시각화
for idx, img in enumerate(imgs):
    plt.figure(figsize=(6, 3))
    plt.imshow(img, cmap='gray')
    plt.title(f'번호판 {idx + 1}')
    plt.axis('off')
    plt.show()
