import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2   

# img = cv2.imread('./data/a1.png')
# cv2.imshow(' 1 title', img)
# cv2.waitKey()

num = int(input('사진을 고르시오:  '))
coin_no =  f'./coin/{num}.png'
img = cv2.imread(coin_no, cv2.IMREAD_COLOR) 
height, width, channel = img.shape 
print('11-04-월요일 img.shape 정보', img.shape)
print('11-04-월요일 width=',width, 'height=', height,  'channel =', channel )
cv2.imshow('11-04 test ', img)
cv2.waitKey()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          
blur = cv2.GaussianBlur(gray, (3,3), 0)    
circle = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 30, None, 200)

if circle is not None:
    circle = np.uint16(np.around(circle))
    
    for k in circle[0, :]:
        cv2.circle(img, (k[0], k[1]), k[2], (0,255,0), 2) 
        cv2.circle(img, (k[0], k[1]), 2, (0,0,255), 5) 


def preprocess_img(coin_no):

    print(f'\n전처리 함수 호출: 동전 번호 {coin_no}')
    img_path = f'./coin/{coin_no:02d}.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {img_path}")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    _, th_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    return img, morph




def find_coins(img, min_radius=25):

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = [cv2.minEnclosingCircle(c) for c in contours]
    return [(tuple(map(int, center)), int(radius)) for center, radius in circles if radius > min_radius]



def extract_coins(src, circles, scale=3):

    coins = []
    for center, radius in circles:
        r = int(radius * scale)
        mask = np.zeros((r, r, 3), np.uint8)
        cv2.circle(mask, (r//2, r//2), radius, (255,255,255), -1)
        coin = cv2.getRectSubPix(src, (r, r), center)
        masked_coin = cv2.bitwise_and(coin, mask)
        coins.append(masked_coin)
    return coins

def classify_coins(circles, hists):

    weights = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,8,6,5,4,3,2,1,0,0,0,0,0,0]
    sim = np.sum(np.array(hists) * np.array(weights), axis=1) / np.sum(hists, axis=1)
    groups = [1 if s > 1.2 else 0 for s in sim]
    
    # 그룹별 반지름에 따른 동전 분류
    g = np.full((2,70), -1, dtype=int)
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3
    
    ncoins = [0]*4
    classes = []
    for group, (_, radius) in zip(groups, circles):
        coin = g[group, radius]
        if 0 <= coin < 4:
            ncoins[coin] +=1
        classes.append(coin)
    return np.array(ncoins), classes

img, mor = preprocess_img(num)
circles = find_coins(mor)
print(f'확인된 WON들: {circles}')

coins = extract_coins(img, circles)
print(f'추출한 동전들: {len(coins)}')

hists = [cv2.calcHist([cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)], [0], None, [32], [0,180]).flatten() for coin in coins]
    
# 동전 분류
ncoins, coin_classes = classify_coins(circles, hists)
print(f'동전 개수: {ncoins}, 동전 종류들: {coin_classes}')

# 총 금액 계산
values = [10,50,100,500]
total = sum(v*c for v,c in zip(values, ncoins))
print(f'TOTAL: {total} WON')

# 결과 그리기
colors = [(0,0,255),(255,255,0),(0,250,0),(255,0,255)]
for (c,r), cls in zip(circles, coin_classes):
    if cls >=0:
        cv2.circle(img, c, r, colors[cls], 2)
        cv2.putText(img, str(values[cls]), (c[0]-15, c[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[3], 2)
cv2.putText(img, f'TOTAL: {total:,} WON', (700,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,230,0), 2)
cv2.imshow('결과', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 동전별 개수 출력
for v, c in zip(values, ncoins):
    print(f'{v}WON: {c}개')
print(f'TOTAL: {total} WON')



