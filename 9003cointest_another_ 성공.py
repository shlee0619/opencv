import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

# 한글 폰트 설정 (윈도우 기준)
font_path = 'c:/windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

def preprocess_image(coin_no):

    print(f'\n전처리 함수 호출: 동전 번호 {coin_no}')
    image_path = f'./coin/{coin_no:02d}.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    _, th_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    return image, morph

def find_coins(image, min_radius=25):

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def main(coin_no=23):

    # 전처리
    image, morph = preprocess_image(coin_no)
    if image is None or morph is None:
        return
    
    # 동전 검출
    circles = find_coins(morph)
    print(f'Detected Circles: {circles}')
    
    if not circles:
        print("동전이 검출되지 않았습니다.")
        return
    
    # 동전 이미지 추출
    coins = extract_coins(image, circles)
    print(f'Extracted Coins: {len(coins)}')
    
    # Hue 히스토그램 계산
    hists = [cv2.calcHist([cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)], [0], None, [32], [0,180]).flatten() for coin in coins]
    
    # 동전 분류
    ncoins, coin_classes = classify_coins(circles, hists)
    print(f'Coin Counts: {ncoins}, Coin Classes: {coin_classes}')
    
    # 총 금액 계산
    values = [10,50,100,500]
    total = sum(v*c for v,c in zip(values, ncoins))
    print(f'Total: {total} Won')
    
    # 결과 그리기
    colors = [(0,0,255),(255,255,0),(0,250,0),(255,0,255)]
    for (c,r), cls in zip(circles, coin_classes):
        if cls >=0:
            cv2.circle(image, c, r, colors[cls], 2)
            cv2.putText(image, str(values[cls]), (c[0]-15, c[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[3], 2)
    cv2.putText(image, f'Total: {total:,} Won', (700,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,230,0), 2)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 동전별 개수 출력
    for v, c in zip(values, ncoins):
        print(f'{v}원: {c}개')
    print(f'Total coin: {total} Won')

if __name__ == "__main__":
    main(coin_no=23)
