import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt
import json
import librosa
import subprocess
import math

# 손가락 관절의 각도 계산 함수
def calculate_angles(landmarks):
    angles = []
    # 각 손가락에 대한 관절 인덱스 정의 (MediaPipe 손 랜드마크 인덱스)
    finger_joints = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    for finger, joints in finger_joints.items():
        # 각 손가락의 관절 각도 계산
        mcp = np.array(landmarks[joints[0]])  # 손가락 시작점 (MCP 관절)
        pip = np.array(landmarks[joints[1]])  # PIP 관절
        dip = np.array(landmarks[joints[2]])  # DIP 관절
        tip = np.array(landmarks[joints[3]])  # 손가락 끝 (TIP)

        # 벡터 계산
        vec1 = mcp - pip
        vec2 = pip - dip
        vec3 = dip - tip

        # 각도 계산
        angle1 = calculate_angle(vec1, vec2)
        angle2 = calculate_angle(vec2, vec3)

        angles.extend([angle1, angle2])
    
    return angles

# 두 벡터 사이의 각도 계산 함수
def calculate_angle(v1, v2):
    # 코사인 법칙을 이용하여 각도 계산
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    # 값의 범위를 [-1, 1]로 제한하여 arccos의 입력 범위에 맞춤
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 비디오 파일 경로
video_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_VoodooChild.mp4'
audio_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_VoodooChild.mp3'

# 오디오 추출 (ffmpeg 필요)
subprocess.call(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path])

# 오디오 분석
y_audio, sr = librosa.load(audio_path)
tempo, beats = librosa.beat.beat_track(y=y_audio, sr=sr)
print(f"Estimated tempo: {tempo} BPM")

# 데이터 저장을 위한 리스트 초기화
data = []

# 얼굴, 손, 자세 인식 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose:

    cap = cv2.VideoCapture(video_path)
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
                x = bboxC.xmin
                y = bboxC.ymin
                width = bboxC.width
                height = bboxC.height
                face_data.append((x, y, width, height))
                # 시각화를 위해 픽셀 좌표 계산
                x_px, y_px = int(x * w), int(y * h)
                width_px, height_px = int(width * w), int(height * h)
                cv2.rectangle(frame, (x_px, y_px), (x_px + width_px, y_px + height_px), (255, 0, 255), 2)

        # 손 인식 결과 처리
        hand_data = []
        hand_angles = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    cx_normalized, cy_normalized = landmark.x, landmark.y
                    landmarks.append([cx_normalized, cy_normalized])
                    # 시각화를 위해 픽셀 좌표 계산
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                hand_data.append(landmarks)
                # 손가락 각도 계산
                angles = calculate_angles(landmarks)
                hand_angles.append(angles)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 자세 인식 결과 처리
        pose_data = {}
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                cx_normalized, cy_normalized = landmark.x, landmark.y
                pose_data[f"pose_{i}"] = [cx_normalized, cy_normalized]
                # 시각화를 위해 픽셀 좌표 계산
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        # 데이터 저장 (프레임별 손 움직임, 손가락 각도, 자세, 얼굴 위치)
        data.append({
            "frame": frame_index,
            "face_data": json.dumps(face_data),
            "hand_data": json.dumps(hand_data),
            "hand_angles": json.dumps(hand_angles),
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
hand_angles_sequences = []
for idx, row in data.iterrows():
    hand_json = row['hand_data']
    angles_json = row['hand_angles']
    if isinstance(hand_json, str) and hand_json != '[]':
        try:
            hand_coords = json.loads(hand_json)
            angles_list = json.loads(angles_json)
            # 첫 번째 손만 사용
            if len(hand_coords) > 0:
                hand_landmarks = hand_coords[0]
                angles = angles_list[0]
                landmark_vector = []
                for coord in hand_landmarks:
                    if isinstance(coord, list) and len(coord) == 2:
                        x, y = coord
                        landmark_vector.extend([x, y])
                if landmark_vector:
                    hand_sequences.append(landmark_vector + angles)
            else:
                hand_sequences.append([0]*42 + [0]*len(angles))
        except Exception as e:
            print(f"Error at index {idx} during json.loads: {e}")
            hand_sequences.append([0]*42 + [0]*len(angles))
    else:
        hand_sequences.append([0]*42 + [0]*len(angles))

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

# 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Bi-directional LSTM 모델 정의 및 학습
model_lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64)),
    Dense(y_train.shape[1])
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss = model_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# 예측 및 시각화
y_pred = model_lstm.predict(X_test)

# 첫 번째 샘플 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test[0], label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Hand Movement and Angles Trajectory')
plt.xlabel('Features')
plt.ylabel('Normalized Values and Angles')
plt.legend()
plt.show()

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
    plt.xlabel("Normalized X Position")
    plt.ylabel("Normalized Y Position")
    plt.show()

# 비디오에 예측 결과 오버레이
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_index >= len(y_pred):
        break

    # 예측된 손 좌표 가져오기
    predicted_coords = y_pred[frame_index][:42]  # 좌표 부분만 사용
    predicted_coords = predicted_coords.reshape(-1, 2)  # (21, 2)

    # 좌표를 픽셀 단위로 변환
    h, w, _ = frame.shape
    for coord in predicted_coords:
        cx, cy = int(coord[0] * w), int(coord[1] * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()


