import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import json

import numpy as np




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
                    landmarks.append([cx, cy])
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                hand_data.append(landmarks)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 자세 인식 결과 처리
        pose_data = {}
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                pose_data[f"pose_{i}"] = [cx, cy]
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        # 데이터 저장 (프레임별 손 움직임, 자세, 얼굴 위치)
        data.append({
            "frame": frame_index,
            "face_data": json.dumps(face_data),
            "hand_data": json.dumps(hand_data),
            "pose_data": json.dumps(pose_data)
        })

        # 결과 프레임 표시
        cv2.imshow('Face, Hand, and Pose Detection', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 프레임 인덱스 증가
        frame_index += 1

cap.release()
cv2.destroyAllWindows()

# 데이터 저장 (CSV 파일로 저장)
df = pd.DataFrame(data)
df.to_csv(r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_analysis.csv', index=False)

# 저장된 CSV 파일을 불러와 분석 시작
data = pd.read_csv(r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_analysis.csv')

# 손 움직임 시퀀스 생성
hand_sequences = []
for idx, row in data.iterrows():
    hand_json = row['hand_data']
    if isinstance(hand_json, str) and hand_json != '[]':
        try:
            hand_coords = json.loads(hand_json)
            # 첫 번째 손만 사용
            if len(hand_coords) > 0:
                hand_landmarks = hand_coords[0]
                landmark_vector = []
                for coord in hand_landmarks:
                    if isinstance(coord, list) and len(coord) == 2:
                        x, y = coord
                        landmark_vector.extend([x, y])
                if landmark_vector:
                    hand_sequences.append(landmark_vector)
            else:
                hand_sequences.append([0]*42)
        except Exception as e:
            print(f"Error at index {idx} during json.loads: {e}")
            hand_sequences.append([0]*42)
    else:
        hand_sequences.append([0]*42)

# numpy 배열로 변환
hand_sequences = np.array(hand_sequences)

# 시퀀스 데이터 생성
sequence_length = 30  # 예: 30 프레임
X_lstm = []
y_lstm = []

for i in range(len(hand_sequences) - sequence_length):
    X_lstm.append(hand_sequences[i:i+sequence_length])
    y_lstm.append(hand_sequences[i+sequence_length])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# LSTM 모델 정의 및 학습
model_lstm = Sequential([
    LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
    Dense(42)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=32)

# 자세 데이터 클러스터링
pose_vectors = []
for idx, pose_json in enumerate(data['pose_data']):
    if isinstance(pose_json, str) and pose_json != '{}':
        try:
            pose_coords = json.loads(pose_json)
            pose_vector = []
            for landmark in pose_coords.values():
                pose_vector.extend(landmark)
            pose_vectors.append(pose_vector)
        except Exception as e:
            print(f"Pose data eval error at index {idx}: {e}")

# 클러스터링 수행 및 시각화
if len(pose_vectors) > 0:
    pose_vectors = np.array(pose_vectors)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pose_vectors)
    pose_clusters = kmeans.labels_
    plt.scatter(pose_vectors[:, 0], pose_vectors[:, 1], c=pose_clusters)
    plt.title("Pose Clustering")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

