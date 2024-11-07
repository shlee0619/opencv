import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from hmmlearn import hmm  # Hidden Markov Model
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
'''
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
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                hand_data.append(landmarks)
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

cap.release()
cv2.destroyAllWindows()

# 데이터 저장 (CSV 파일로 저장)
df = pd.DataFrame(data)
df.to_csv(r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_analysis.csv', index=False)
'''
# 저장된 CSV 파일을 불러와 분석 시작
data = pd.read_csv(r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_analysis.csv')

# 손, 얼굴, 자세 데이터를 시간 순서대로 준비
hand_data = data['hand_data']
pose_data = data['pose_data']

# 유효한 hand_data 항목이 있는지 확인
valid_hand_data = data[data['hand_data'] != '[]']
print(f"Number of valid hand_data entries: {len(valid_hand_data)}")


'''
# 손 움직임 시퀀스 생성
hand_movement_sequences = []

for idx, row in valid_hand_data.iterrows():
    hand = row['hand_data']
    hand_sequence = []
    if isinstance(hand, str) and hand != '[]':
        try:
            # 문자열을 리스트로 변환
            hand_coords = eval(hand)
            if isinstance(hand_coords, list) and all(isinstance(coord, list) for coord in hand_coords):
                for coord in hand_coords:
                    if len(coord) >= 2:
                        x, y = coord[0], coord[1]
                        hand_sequence.append([x, y])
        except Exception as e:
            print(f"Error at index {idx} during eval: {e}")
    
    # 빈 시퀀스가 아닌 경우에만 추가
    if len(hand_sequence) > 0:
        hand_movement_sequences.append(np.array(hand_sequence))

# 모든 시퀀스를 같은 길이로 맞추기 위해 패딩 처리
padded_sequences = pad_sequences(hand_movement_sequences, padding='post', dtype='float32')

# LSTM 모델 정의 및 학습
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(padded_sequences.shape[1], 2)),
    LSTM(50),
    Dense(2)
])

# X_lstm, y_lstm 생성
X_lstm = padded_sequences[:-1]
y_lstm = padded_sequences[1:]

# y_lstm의 차원을 X_lstm과 일치시킴
y_lstm = y_lstm[:, :X_lstm.shape[1], :]

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=10)
'''
'''
# 손 움직임 시퀀스 생성
hand_movement_sequences = []

for idx, row in valid_hand_data.iterrows():
    hand = row['hand_data']
    hand_sequence = []
    if isinstance(hand, str) and hand != '[]':
        try:
            # 문자열을 리스트로 변환
            hand_coords = eval(hand)
            if isinstance(hand_coords, list) and all(isinstance(coord, list) for coord in hand_coords):
                for coord in hand_coords:
                    if len(coord) >= 2:
                        x, y = coord[0], coord[1]
                        hand_sequence.append([x, y])
        except Exception as e:
            print(f"Error at index {idx} during eval: {e}")
    
    # 빈 시퀀스가 아닌 경우에만 추가하고, 중첩되지 않은 배열로 저장
    if len(hand_sequence) > 0:
        hand_movement_sequences.append(np.array(hand_sequence).reshape(-1, 2))  # (n, 2) 형태로 저장

# valid_sequences의 모든 시퀀스를 동일한 길이로 패딩 처리
padded_sequences = pad_sequences(hand_movement_sequences, padding='post', dtype='float32')

# LSTM 모델 정의 및 학습
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(padded_sequences.shape[1], 2)),
    LSTM(50),
    Dense(2)
])

# X_lstm, y_lstm 생성
X_lstm = padded_sequences[:-1]
y_lstm = padded_sequences[1:]

# y_lstm의 타임스텝 길이를 X_lstm과 일치하도록 맞춤
y_lstm = y_lstm[:, :X_lstm.shape[1], :]

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=10)
'''

# 손 움직임 시퀀스 생성
hand_movement_sequences = []

for idx, row in data.iterrows():
    hand = row['hand_data']
    hand_sequence = []
    if isinstance(hand, str) and hand != '[]':
        try:
            # 문자열을 리스트로 변환
            hand_coords = eval(hand)
            if isinstance(hand_coords, list) and all(isinstance(coord, list) for coord in hand_coords):
                # 모든 좌표가 (x, y) 형태인지 확인
                for coord in hand_coords:
                    if len(coord) == 2:
                        x, y = coord
                        hand_sequence.append([x, y])
        except Exception as e:
            print(f"Error at index {idx} during eval: {e}")

    # 빈 시퀀스가 아닌 경우에만 추가하고, (n, 2) 형태로 저장
    if hand_sequence:
        hand_movement_sequences.append(np.array(hand_sequence))

# 모든 시퀀스를 같은 길이로 맞추기 위해 패딩 처리
padded_sequences = pad_sequences(hand_movement_sequences, padding='post', dtype='float32')

# LSTM 모델 정의
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(padded_sequences.shape[1], 2)),
    LSTM(50),
    Dense(2)
])

# X_lstm과 y_lstm 생성
X_lstm = padded_sequences[:-1]
y_lstm = padded_sequences[1:]

# y_lstm의 타임스텝 길이를 X_lstm과 일치시킴
y_lstm = y_lstm[:, :X_lstm.shape[1], :]

# 모델 학습
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=10)

# 자세 데이터 클러스터링
pose_vectors = []
for pose in pose_data:
    if isinstance(pose, str) and pose != '{}':
        try:
            pose_coords = eval(pose)
            pose_vector = [coord for landmark in pose_coords.values() for coord in landmark]
            pose_vectors.append(pose_vector)
        except Exception as e:
            print(f"Pose data eval error: {e}")

# 클러스터링 수행 및 시각화
if len(pose_vectors) > 0:
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pose_vectors)
    pose_clusters = kmeans.labels_
    plt.scatter(np.array(pose_vectors)[:, 0], np.array(pose_vectors)[:, 1], c=pose_clusters)
    plt.title("Pose Clustering")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()