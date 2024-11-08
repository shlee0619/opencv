import cv2
import numpy as np
import matplotlib.pyplot as plt


# 이미지 번호 설정
car_no = 5  # 처리할 이미지 번호 (05, 13, 15, 20 등)

def preprocessing(car_no):
    """이미지 전처리: 그레이스케일 변환, 블러링, 엣지 검출, 이진화, 형태학적 변환"""
    image_path = f'./images/car/{car_no:02d}.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 '{image_path}'에서 찾을 수 없습니다.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edged = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
    
    _, thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 13), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return image, morph

def car_size(size, area_range=(3000, 12000), ratio_range=(2.0, 6.5)):
    """번호판 후보의 크기와 비율을 검사"""
    w, h = size
    if w == 0 or h == 0:
        return False
    ratio = h / w if h > w else w / h
    area = w * h
    return area_range[0] < area < area_range[1] and ratio_range[0] < ratio < ratio_range[1]

def get_contours(image):
    """OpenCV 버전에 따른 Contour 반환값 처리"""
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def find_license_plate_candidates(morph):
    """번호판 후보 Contour 찾기 및 필터링"""
    contours, _ = get_contours(morph)
    candidates = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        center, size, angle = rect
        if car_size(size):
            candidates.append(rect)
    return candidates

def extract_characters(image, candidates):
    """번호판에서 문자 추출"""
    characters = []
    for rect in candidates:
        center, size, angle = rect
        if size[0] < size[1]:
            angle -= 90
            size = (size[1], size[0])
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        cropped = cv2.getRectSubPix(rotated, (int(size[0]), int(size[1])), center)
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_cropped, (144, 28))
        characters.append(resized)
    return characters

def visualize_image(title, image, cmap_type='gray'):
    """이미지 시각화"""
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 전처리 수행
image, morph = preprocessing(car_no)
visualize_image('Preprocessing 결과 (Morph)', morph)

# 번호판 후보 찾기
candidates = find_license_plate_candidates(morph)
print(f"번호판 후보 수: {len(candidates)}")

if not candidates:
    print("번호판 후보가 검출되지 않았습니다.")
    exit()

# 번호판에서 문자 추출
characters = extract_characters(image, candidates)
print('자동차 번호만 추출 성공')

# 추출된 번호판 이미지 시각화 및 저장
for idx, char_img in enumerate(characters, 1):
    visualize_image(f'번호판 {idx}', char_img)
    save_path = f'./images/car/number_plate_{car_no:02d}_{idx}.png'
    cv2.imwrite(save_path, char_img)
    print(f"번호판 {idx} 이미지를 '{save_path}'로 저장했습니다.")
