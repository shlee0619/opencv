import cv2  #pip install opencv-python



# 영상 검출기, 재생
def videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list):

    while True: #라이브로 들어오는 비디오를 프레임별로 캡쳐해서 화면에 디스플레이 해야함 고로 while 문에서 키를 누를떄까지 무한루프

        # 캡처 이미지 불러오기  비디오 프레임을 제대로 읽으면 ret값 true 아니라면 false
        ret,img = cam.read()
        # 영상 압축, 이미지 크기 변경
        try:
            img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0) #fx, fy 에 계산된 크기가 dsize에 할당
        except: break #아니라면 종료
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                           minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )

        for box in results:  #탐지된 객체의 경계박스 리스트
            x, y, w, h = box  #예측된 얼굴의 위치
            face = img[int(y):int(y+h),int(x):int(x+h)].copy() #얼굴 이미지 추출
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #성별 및 연령 예측을 위한 이미지 변환 및 전처리
            '''
            face = 입력 영상,
            scalefactor = 입력 영상 픽셀값에 곱할 값 : 기본값은 1
            size = 출력영상 크기 (227, 227)
            Model_MEAN_VALUES = 입력영상 각 채널에서 뺄 평균값 기본값은 (0, 0, 0, 0)
            swapRB = R 과 B 채널을 서로 바꿀 것인지를 결정하는 플래그, 기본값은 False
            '''


            # gender detection      /// net.setInput함수로 네트워크 입력 설정하기
            gender_net.setInput(blob) #
            gender_preds = gender_net.forward() #gender_net 추론을 진행할때 사용하는 함수
            gender = gender_preds.argmax()      #네트워크 순방향으로 실행 argmax()적용하면 가장 인덱스에 근접한 값
            # Predict age
            age_net.setInput(blob) #age_net을 setInput 함수로 네트워크 입력 설정
            age_preds = age_net.forward() #age_net 네트워크 순방향으로 짆행
            age = age_preds.argmax()      #argmax()로 가장 인덱스에 근접한 값
            
            info = gender_list[gender] +' '+ age_list[age] #정보를 나타냄 성별 리스트와 나이 리스트

            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2) #2의 두께의 선으로 직사각형 이미지 생성
            cv2.putText(img,info,(x,y-15),0, 0.5, (0, 255, 0), 1) #텍스트의 위치, 직사각형 이미지 위에 성별과 나이 예측


         # 영상 출력
        cv2.imshow('facenet',img)

        if cv2.waitKey(1) > 0: #1일때 영상을 연속해서 끊기지않게 출력
            break

# # 사진 검출기
def imgDetector(img,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list):
    # 영상 압축
    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade 얼굴 탐지 알고리즘
    results = cascade.detectMultiScale(gray,            # 입력 이미지
                                       scaleFactor= 1.5,# 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20,20)  # 탐지 객체 최소 크기
                                       )

    for box in results:
        x, y, w, h = box
        face = img[int(y):int(y+h),int(x):int(x+h)].copy()
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_preds.argmax()
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds.argmax()
        info = gender_list[gender] +' '+ age_list[age]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
        cv2.putText(img,'face test',(50,50), 1,1, (255,0,0), 2, cv2.LINE_4)
        
    # 사진 출력
    cv2.imshow('facenet',img)
    cv2.waitKey(5000) #cv2.waitKey(10000)




# 얼굴 탐지 모델 가중치
cascade_filename = './data/haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename) #학습 모델 불러오기

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_net = cv2.dnn.readNetFromCaffe(
	'./data/deploy_age.prototxt',
	'./data/age_net.caffemodel') #age_net 모델 불러오기

gender_net = cv2.dnn.readNetFromCaffe(
	'./data/deploy_gender.prototxt',
	'./data/gender_net.caffemodel') #gender_net 모델 불러오기

age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)'] #나이의 리스트
gender_list = ['Male', 'Female'] #성별 구분


# 영상 파일 
cam = cv2.VideoCapture('./data/sample.mp4')
videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list )


