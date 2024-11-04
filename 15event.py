import cv2
import sys

title = 'mouse event'                  
img = cv2.imread('./data/blank_500.jpg') 
cv2.imshow('mouse event', img)                  

colors={
         'black':(0,0,0),    'red' : (255,0,0),
         'blue':(0,0,255),    'green': (0,255,0)
        }

def onMouse(event, x, y, flags, param):    
    print('좌표값 ' ,  event, x, y, flags, param)   
    color = colors['black']
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누름인 경우 
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY : 
            color = colors['green']
        elif flags & cv2.EVENT_FLAG_SHIFTKEY :
            color = colors['blue']
        elif flags & cv2.EVENT_FLAG_CTRLKEY : 
            color = colors['red']
        
        cv2.circle(img, (x,y),50, color, -1)  # 지름 50 크기의 검은색 원을 해당 좌표에 그림
        cv2.imshow(title, img)                # 그려진 이미지를 다시 표시 


cv2.setMouseCallback(title, onMouse)    # GUI 윈도우에 등록 

if cv2.waitKey(0) & 0xFF == 27:     
    print('프로그램을 종료합니다 ')
    cv2.destroyAllWindows()
    sys.exit()    
