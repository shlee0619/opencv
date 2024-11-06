import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2   


# 함수preprocessing(coin_no)기술 
def preprocessing(coin_no):
    print('\n11-05-화요일 def preprocessing(coin_no): 전처리 함수호출 ')
    image = cv2.imread('./coin/%02d.png' %coin_no, cv2.IMREAD_COLOR)
    print(image)
    print()
    if image is None:
        return None, None
    cv2.imshow('def preprocessing(coin_no): ', image)
    cv2.waitKey()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7,7), 2, 2)              # 블러링
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU              # 오츠(otsu) 이진화 지정
    th_one, th_img = cv2.threshold(gray, 130, 255, flag)    # 이진화
    print('th_one 결과' , th_one)

    mask = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask) # morphologyEx 연산
    return image, morph

## 3번째 testing ... 
coin_no = 23  # 1 3 5 7 
image, morph = preprocessing(coin_no)
print()
print( '동전 리턴값 image =' , image  )
print('- ' * 50)
print( '동전 리턴값 morph =' , morph)
print()


#----------------------------------------------------------------------------------------------
## 함수find_coins(image)기술 
def find_coins(image):
    print('\n11-05-화요일 def find_coins(image): 함수호출 ')
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    circles = [cv2.minEnclosingCircle(c) for c in contours] # 외각 감싸는 원 검출
    circles = [(tuple(map(int, center)), int(radius))
               for center, radius in circles if radius>25]
    return circles


circles = find_coins(morph)
print(circles)
print()

#----------------------------------------------------------------------------------------------
## 함수make_coin_img(src, circles)기술 
def make_coin_img(src, circles):
    print('11-05-화요일 make_coin_img(src, circles): 함수호출 ')
    coins = []
    for center, radius in circles:
        r = radius * 3                                      # 검출 동전 반지름 3배
        cen = (r // 2, r // 2)                              # 마스크 중심
        mask = np.zeros((r,r,3), np.uint8)                  # 마스크 행렬
        cv2.circle(mask, cen, radius, (255,255,255), cv2.FILLED)
        coin = cv2.getRectSubPix(src, (r,r), center)        # 동전 영상 가져오기
        coin = cv2.bitwise_and(coin, mask)                  # 마스킹 처리
        coins.append(coin)                                  # 동전 영상 저장장
    return coins


coin_imgs = make_coin_img(image, circles)
print(coin_imgs) #0으로 출력
print()

#----------------------------------------------------------------------------------------------
## 함수grouping(hists)기술 
def calc_histo_hue(coin):
    print('11-05-화요일 calc_histo_hue(coin): 함수호출 ')
    hsv = cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)             # 컬러 공간 변환
    hsize, ranges = [32], [0,180]                           # 32개 막대, 화소값 0~180 범위
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)    # 0(Hue)채널 히스토그램 계산
    return hist.flatten()                                   # 1차원 전개 후 반환

coin_hists = [calc_histo_hue(coin)  for coin in coin_imgs]
print(coin_hists)
print(': ' * 50)
print()

#----------------------------------------------------------------------------------------------
## 함수grouping(hists)기술 
def grouping(hists):
    print('grouping(hists): 함수호출 ')
    ws = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
          4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]   # 가중치 32개 원소 지정

    sim = np.multiply(hists, ws)                            # 히스토그램과 가중치 곱
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)   # 가중치 곱의 합/히스토그램 합
    groups = [1 if s > 1.2 else 0 for s in similaritys]

    ## 결과 보기
    x = np.arange(len(ws))                             
    for i, s in enumerate(similaritys):
          print('11-05-화요일 %d %5.0f  %d' % (i, s, groups[i]))
    return groups


groups = grouping(coin_hists)
print(groups) #[1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
print()


#----------------------------------------------------------------------------------------------
## 함수classify_coins(circles, groups)기술 
def classify_coins(circles, groups):
    print('11-05-화요일 classify_coins(circles, groups): 함수호출 ')
    ncoins = [0] * 4
    coin_class = []
    g = np.full((2,70), -1, np.int_)                         # 2행으로 두 개 그룹 설정      
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3           # 10원 그룹- 10원 가능성 확대
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3           # 50원 그룹- 50원 100원 가능성 확대

    for group, (_, radius) in zip(groups, circles):         # 동전 객체 순회
        coin = g[group, radius]                             # 동전 종류 확정
        coin_class.append(coin)
        ncoins[coin] += 1                                   # 동전별 개수 산정
    return np.array(ncoins), coin_class                     # 넘파이 행렬로 반환


ncoins, coin_class = classify_coins(circles, groups) 
print(ncoins, coin_class)
print()


tens= ncoins[0]*10
fiftys= ncoins[1]*50
baks= ncoins[2]*100
obaks= ncoins[3]*500
total = tens + fiftys + baks + obaks
print('11-05-화요일 tens =',tens)
print('11-05-화요일 fiftys =',fiftys)
print('11-05-화요일 baks =',baks)
print('11-05-화요일 obaks =',obaks)
print('11-05-화요일 total =',total)      #[0]= 10 [1]=50 [2]=100 [3]=500


color = [(0,0,255),(255,255,0), (0,250,0), (255,0,255) ]
for i, (c,r) in enumerate(circles):
    cv2.circle(image, c, r, color[coin_class[i]], 2)


cv2.imshow('circle testing ', image)
cv2.waitKey()
cv2.destroyAllWindows()


#----------------------------------------------------------------------------------------------
#숫자화 총금액 산출
coin_value = np.array([10,50,100,500])
for i in range(4):
    print('%3d원: %3d개' % (coin_value[i], ncoins[i]))      # 동전별 개수 출력
total = sum(coin_value * ncoins)                            # 동전 금액 * 동전별 개수
string = 'Total coin: {:,} Won'.format(total)               # 계산 금액 문자열
print(string)  #계산결과  
'''
 10원:  14개
 50원:   2개
100원:   6개
500원:   5개
Total coin: 3,340 Won
'''


#----------------------------------------------------------------------------------------------
## 함수put_string(image, text, pt, value, color=(120,200,90))기술
def put_string(image, text, pt, value, color=(120,200,90)):     #문자열 출력 함수
    text += str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, pt, font, 0.7, color, 3)           #글자 적기

put_string(image, string, (700,50), '', (0,230,0))              #영상에 출력

# color = [(0,0,250), (255,255,0), (0,250,0), (250,0,255)]    # 위에서 기술되어 있음
for i, (c, r) in enumerate(circles):
    cv2.circle(image, c, r, color[coin_class[i]], 2)
    put_string(image, str(coin_value[coin_class[i]]), (c[0]-15, c[1]+15), '', color[3])     # 동전반지름

cv2.imshow('11-05 result image', image)
cv2.waitKey(0)


print()