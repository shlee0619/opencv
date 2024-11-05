import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


path = './images/car/05.jpg'
img_origin = cv2.imread(path, cv2.IMREAD_COLOR) 
gray = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HSV)
plt.title('cv2.COLOR_RGB2HSV')  
plt.imshow(gray,cmap='gray')
plt.show()



#적응 임계처리 pDF문서 71페이지
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


# 가우시안블러 - 그림의 노이즈 제거 
# Contours를 찾으려면 검은색 배경에 흰색 바탕이어야 함
# Contours란 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
# GauusianBlur적용해서 Thresholding 글자번호추출
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

plt.title('cv2.GaussianBlur 11-05')
plt.imshow(img_blur_thresh, cmap='gray')
plt.show()
print()

##########################################################################################################
##########################################################################################################
print('11-05-화요일  GaussianBlur  findContours함수')

#순서0]  c preprocessing(car_no)함수
def preprocessing(car_no):
    print('\n11-05-화요일 def preprocessing(car_no): 전처리 함수호출 ')
    image = cv2.imread('./images/car/%02d.jpg' %car_no, cv2.IMREAD_COLOR)

    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 2, 2)   
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0,3)

    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU              # 오츠(otus) 이진화 지정
    th_one, th_img = cv2.threshold(gray, 130, 255, flag)    # 이진화
    kernel = np.ones( (5,13), np.uint8) #마스크용
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return image, morph



#순서1]  car_size(size)함수
def car_size(size):  #사용호출 csa = [~~~,size,angle in rects if car_size(size)]
    print('\n11-05-화요일 car_size(size): ')
    w,h = size
    data = 0
    if w==0 or h==0 :
        return False
    
    if h>w:
        data = h/w
    else: 
        data = w/h
  
    chk1 = 3000 < (h*w) < 12000 #번호판 넓이 조건
    chk2 = 2.0 < data  < 6.5 
    print('car_size 함수 chk1 =', chk1, '  chk2 =',chk2) # False True 불값출력
    return (chk1 and chk2)


#순서2]  car_find(morph)함수
def car_find(morph):
    print('\n11-05-화요일car_find(morph): ')
    result = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    print()
    rects = [ cv2.minAreaRect(ct) for ct in contours ] 
    csa = [(tuple(map(int,center)), tuple(map(int, size)), angle)  for center,size,angle in rects if car_size(size)]
    for center,size,angle in rects:
        print('center =', center, '  size =', size , ' angle =', angle)

    plt.figure(figsize=(10,6))
    plt.imshow(morph, cmap='gray')
    plt.title('car_find(morph) 1111 ')
    plt.show()
    return csa 

# path = './images/car/05.jpg'
car_no = 5  # 5, 13, 15, 20 // 61어 1099번호
image, morph = preprocessing(car_no) #데이터회색,블러
mycar = car_find(morph) #이미지컨투어 사각형 지정 



#순서3]  car_charnumber(image, size)함수
def car_charnumber(image, center):
    print('11-05-화요일 car_charnumber(image, center)함수')
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)           # 채움 영역
    dif1, dif2 = (25, 25, 25), (25, 25, 25)             # 채움 색상 범위
    flags = 4 + 0xff00 + cv2.FLOODFILL_FIXED_RANGE      # 채움 방향
    flags += cv2.FLOODFILL_MASK_ONLY

    # 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint( -15, 15, (20,2) )          # 20개 좌표 생성
    pts = pts + center
    for x, y in pts:                                     # 랜덤 좌표 평행 이동
        if 0 <= x < w and 0 <= y < h:
            a, b, fill, c = cv2.floodFill(image, fill, (x,y), 255, dif1, dif2, flags)
            print('a =',a ,'  b =', b , '  fill =' , fill, '  c =',c  )

    # 이진화 및 외곽영역 추출사각형 검출
    return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]


fills = [ car_charnumber(image, size) for size, a, b in mycar]
print()
for size, a, b in mycar:
    print('size =' , size, '  a =', a  ,' b=', b)


#순서4] car_position(image, k)함수
def car_position(image, rect):
    print('\n11-05-화요일 car_position(image, rect)함수')
    center, (w,h), angle = rect 

    if w<h:
        w,h = h,w
        angle = angle-90

    size = image.shape[1::-1]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)
    crop_img =  cv2.getRectSubPix(rot_img, (w,h), center)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10,6))
    plt.imshow(crop_img , cmap='gray') 
    plt.title('final  car_position(image, rect)')
    plt.show()
    return cv2.resize(crop_img, (144,28))


new_cd = [car_find(fill) for fill in fills]
new_cd = [cand[0] for cand in new_cd  if cand] 
imgs = [car_position(image, k) for k in new_cd]
print('자동차 번호만 추출 성공 ')
