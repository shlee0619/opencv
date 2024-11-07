import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 비디오 파일 경로
video_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_Acoustic.mp4'
cap = cv2.VideoCapture(video_path)

# 데이터 저장을 위한 리스트 초기화
data = []

# 얼굴, 손, 자세 인식 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose:

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 인식
        face_results = face_detection.process(rgb_frame)

        # 손 인식
        hand_results = hands.process(rgb_frame)

        # 자세 인식
        pose_results = pose.process(rgb_frame)

        # 프레임의 높이와 너비 가져오기
        h, w, _ = frame.shape

        # 얼굴 인식 결과 처리
        face_data = []
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                face_data.append((x, y, width, height))
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 2)

        # 손 인식 결과 처리
        hand_data = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 손가락 점 표시
                hand_data.append(landmarks)

                # 손 랜드마크를 연결하여 그리기
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 자세 인식 결과 처리
        pose_data = {}
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                pose_data[f"pose_{i}"] = (cx, cy)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        # 데이터 저장 (프레임별 손 움직임, 자세, 얼굴 위치)
        data.append({
            "frame": frame_index,
            "face_data": face_data,
            "hand_data": hand_data,
            "pose_data": pose_data
        })

        # 결과 프레임 표시
        cv2.imshow('Face, Hand, and Pose Detection', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 프레임 인덱스 증가
        frame_index += 1

# 비디오와 창 닫기
cap.release()
cv2.destroyAllWindows()

# 데이터 저장 (CSV 또는 JSON)
df = pd.DataFrame(data)
df.to_csv(r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_analysis.csv', index=False)
