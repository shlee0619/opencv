import cv2
import face_recognition
import mediapipe as mp

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# 비디오 파일 경로
video_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_VoodooChild.mp4'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 얼굴 및 손 인식 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 인식
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                # 사각형 좌표 계산
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 2)

        # 손 인식
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 손가락 점 표시

                # 손 랜드마크를 연결하여 그리기
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 결과 프레임 표시
        cv2.imshow('Face and Hand Detection', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
