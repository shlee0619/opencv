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
import os

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

# 기타 지판 영역 정의 함수
def get_guitar_fretboard_region(frame):
    # 실제로는 기타 위치를 자동으로 탐지해야 하지만, 여기서는 간단히 고정된 영역으로 가정합니다.
    h, w, _ = frame.shape
    x_start = int(w * 0.2)
    x_end = int(w * 0.6)
    y_start = int(h * 0.3)
    y_end = int(h * 0.9)
    return x_start, y_start, x_end, y_end

# 지판을 프렛과 줄로 나누는 함수
def get_fretboard_grid(x_start, y_start, x_end, y_end, num_frets=12, num_strings=6):
    fret_positions = np.linspace(x_start, x_end, num=num_frets+1).astype(int)
    string_positions = np.linspace(y_start, y_end, num=num_strings).astype(int)
    return fret_positions, string_positions

# 비디오 및 오디오 파일 경로
video_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_Acoustic.mp4'
audio_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_Acoustic.mp3'

# 오디오 추출 (ffmpeg 필요)
if not os.path.exists(audio_path):
    subprocess.call(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path])
else:
    print("Audio file already exists.")

# 오디오 분석
y_audio, sr = librosa.load(audio_path)
tempo, beats = librosa.beat.beat_track(y=y_audio, sr=sr)
print(f"Estimated tempo: {tempo} BPM")

# 추가적인 오디오 특징 추출
# 스펙트럼 특징 추출
spectrogram = np.abs(librosa.stft(y_audio))
spectral_centroid = librosa.feature.spectral_centroid(S=spectrogram, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spectrogram, sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(S=spectrogram, sr=sr)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y_audio)

# 온셋 감지
onset_env = librosa.onset.onset_strength(y=y_audio, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# 데이터 저장을 위한 리스트 초기화
data = []

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 얼굴, 손, 자세 인식 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 속도 얻기
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴, 손, 자세 인식
        face_results = face_detection.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        # 프레임의 높이와 너비 가져오기
        h, w, _ = frame.shape

        # 기타 지판 영역 설정 및 시각화
        x_start, y_start, x_end, y_end = get_guitar_fretboard_region(frame)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)

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
        finger_positions = []  # 모든 손에 대한 손가락 위치
        if hand_results.multi_hand_landmarks:
            fret_positions, string_positions = get_fretboard_grid(x_start, y_start, x_end, y_end)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = []
                hand_finger_positions = {}  # 현재 손의 손가락 위치
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    cx_normalized, cy_normalized = landmark.x, landmark.y
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([cx_normalized, cy_normalized])
                    # 지판 영역 내에 있는지 확인
                    if x_start <= cx <= x_end and y_start <= cy <= y_end:
                        # 프렛과 줄 결정
                        fret = int(np.searchsorted(fret_positions, cx) - 1)
                        string = int(np.searchsorted(string_positions, cy))
                        hand_finger_positions[idx] = {'fret': fret, 'string': string}
                        # 시각화
                        cv2.putText(frame, f"F{fret}S{string}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    else:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                hand_data.append(landmarks)
                finger_positions.append(hand_finger_positions)
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

        # 현재 프레임의 시간 계산
        frame_timestamp = frame_index / fps

        # 해당하는 오디오 샘플 인덱스 계산
        audio_sample_index = int(frame_timestamp * sr)

        # 현재 프레임의 오디오 특징 추출 (MFCC)
        current_mfcc = librosa.feature.mfcc(y=y_audio[audio_sample_index:audio_sample_index+int(sr/fps)], sr=sr, n_mfcc=13)
        current_mfcc_mean = np.mean(current_mfcc, axis=1).tolist()

        # 온셋 여부 확인
        current_time = frame_timestamp
        onset_detected = any(abs(onset_time - current_time) < (1/fps) for onset_time in onset_times)
        if onset_detected:
            cv2.putText(frame, "Note Onset", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 데이터 저장
        data.append({
            "frame": frame_index,
            "face_data": json.dumps(face_data),
            "hand_data": json.dumps(hand_data),
            "hand_angles": json.dumps(hand_angles),
            "finger_positions": json.dumps(finger_positions),
            "pose_data": json.dumps(pose_data),
            "mfcc": json.dumps(current_mfcc_mean),
            "onset_detected": onset_detected
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

# 필요한 변수 설정
num_landmark_features = 42  # 21 랜드마크 * 2 좌표
num_angle_features = 10     # 5 손가락 * 2 각도
num_mfcc_features = 13      # MFCC 특징 수
num_other_features = 1      # onset_detected
total_features = num_landmark_features + num_angle_features + num_mfcc_features + num_other_features

# 손 움직임 시퀀스 생성
hand_sequences = []
for idx, row in data.iterrows():
    hand_json = row['hand_data']
    angles_json = row['hand_angles']
    mfcc_json = row['mfcc']
    onset_detected = int(row['onset_detected'])  # Boolean을 int로 변환
    try:
        hand_coords = json.loads(hand_json)
        angles_list = json.loads(angles_json)
        mfcc_features = json.loads(mfcc_json)
        # 첫 번째 손만 사용
        if len(hand_coords) > 0 and len(angles_list) > 0:
            hand_landmarks = hand_coords[0]
            angles = angles_list[0]
            landmark_vector = []
            for coord in hand_landmarks:
                if isinstance(coord, list) and len(coord) == 2:
                    x, y = coord
                    landmark_vector.extend([x, y])
            # 각 특징의 길이를 확인하고 패딩 또는 자르기
            landmark_vector = (landmark_vector + [0]*num_landmark_features)[:num_landmark_features]
            angles = (angles + [0]*num_angle_features)[:num_angle_features]
            mfcc_features = (mfcc_features + [0]*num_mfcc_features)[:num_mfcc_features]
            feature_vector = landmark_vector + angles + mfcc_features + [onset_detected]
            hand_sequences.append(feature_vector)
        else:
            # 손 데이터가 없는 경우
            feature_vector = [0]*total_features
            feature_vector[-1] = onset_detected
            hand_sequences.append(feature_vector)
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        # 예외 발생 시 기본값 사용
        feature_vector = [0]*total_features
        feature_vector[-1] = onset_detected
        hand_sequences.append(feature_vector)

# numpy 배열로 변환
hand_sequences = np.array(hand_sequences)

# 시퀀스 데이터 생성
sequence_length = 30  # 예: 30 프레임
X_lstm = []
y_lstm = []

for i in range(len(hand_sequences) - sequence_length):
    X_lstm.append(hand_sequences[i:i+sequence_length])
    y_lstm.append(hand_sequences[i+sequence_length][:num_landmark_features + num_angle_features])  # 손 랜드마크와 각도만 예측

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

# 비디오에 예측 결과 오버레이
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0
test_frame_indices = np.arange(len(X_test)) + sequence_length  # 실제 프레임 인덱스 조정

while True:
    ret, frame = cap.read()
    if not ret or frame_index >= test_frame_indices[-1]:
        break

    if frame_index in test_frame_indices:
        idx = np.where(test_frame_indices == frame_index)[0][0]
        # 예측된 손 좌표 가져오기
        predicted_coords = y_pred[idx][:num_landmark_features]
        predicted_coords = predicted_coords.reshape(-1, 2)  # (21, 2)

        # 좌표를 픽셀 단위로 변환
        h, w, _ = frame.shape
        for coord in predicted_coords:
            x_norm, y_norm = coord
            cx, cy = int(x_norm * w), int(y_norm * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()

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

# ====================== 추가된 시각화 코드 시작 ======================

import seaborn as sns
import matplotlib.animation as animation

# 1. 프렛 및 줄 사용 빈도 히트맵 생성
num_frets = 12
num_strings = 6
fret_string_usage = np.zeros((num_strings, num_frets))

# 손가락 위치 데이터에서 프렛 및 줄 사용 빈도 계산
for idx, row in data.iterrows():
    finger_positions_json = row['finger_positions']
    if isinstance(finger_positions_json, str) and finger_positions_json != '[]':
        try:
            finger_positions_list = json.loads(finger_positions_json)
            for hand_finger_positions in finger_positions_list:
                for finger_idx, pos in hand_finger_positions.items():
                    fret = pos['fret']
                    string = pos['string']
                    if 0 <= fret < num_frets and 0 <= string < num_strings:
                        fret_string_usage[string, fret] += 1
        except Exception as e:
            print(f"Error at index {idx}: {e}")

# 히트맵 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(fret_string_usage, annot=True, fmt=".0f", cmap="Blues")
plt.title("Fretboard Usage Heatmap")
plt.xlabel("Frets")
plt.ylabel("Strings")
plt.xticks(np.arange(num_frets)+0.5, np.arange(num_frets))
plt.yticks(np.arange(num_strings)+0.5, np.arange(num_strings))
plt.gca().invert_yaxis()  # 기타 줄 순서에 맞게 y축 반전
plt.show()

# 2. 손가락 각도 변화 그래프
hand_angles_over_time = []

for idx, row in data.iterrows():
    hand_angles_json = row['hand_angles']
    if isinstance(hand_angles_json, str) and hand_angles_json != '[]':
        try:
            angles_list = json.loads(hand_angles_json)
            # 첫 번째 손만 사용
            if len(angles_list) > 0:
                angles = angles_list[0]
                hand_angles_over_time.append(angles)
            else:
                hand_angles_over_time.append([0]*10)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            hand_angles_over_time.append([0]*10)
    else:
        hand_angles_over_time.append([0]*10)

hand_angles_over_time = np.array(hand_angles_over_time)

# 손가락 각도 그래프 시각화
plt.figure(figsize=(12, 6))
for i in range(hand_angles_over_time.shape[1]):
    plt.plot(hand_angles_over_time[:, i], label=f'Angle {i+1}')
plt.title('Hand Angles Over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.show()

# 3. MFCC 오디오 특징 변화 그래프
mfcc_over_time = []

for idx, row in data.iterrows():
    mfcc_json = row['mfcc']
    if isinstance(mfcc_json, str):
        try:
            mfcc_features = json.loads(mfcc_json)
            mfcc_over_time.append(mfcc_features)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            mfcc_over_time.append([0]*num_mfcc_features)
    else:
        mfcc_over_time.append([0]*num_mfcc_features)

mfcc_over_time = np.array(mfcc_over_time)

# MFCC 특징 그래프 시각화
plt.figure(figsize=(12, 6))
for i in range(mfcc_over_time.shape[1]):
    plt.plot(mfcc_over_time[:, i], label=f'MFCC {i+1}')
plt.title('MFCC Features Over Time')
plt.xlabel('Frame')
plt.ylabel('MFCC Coefficient')
plt.legend()
plt.show()

# 4. 손 움직임과 오디오 특징 간의 상관관계 분석
# 손가락 각도와 MFCC 특징을 결합
combined_features = np.hstack((hand_angles_over_time, mfcc_over_time))
correlation_matrix = np.corrcoef(combined_features.T)

# 상관관계 매트릭스 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix between Hand Angles and MFCC Features')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.show()

# 5. 손 랜드마크 움직임 애니메이션 생성
# 손 랜드마크 데이터 추출
hand_landmarks_over_time = []

for idx, row in data.iterrows():
    hand_json = row['hand_data']
    if isinstance(hand_json, str) and hand_json != '[]':
        try:
            hand_coords = json.loads(hand_json)
            # 첫 번째 손만 사용
            if len(hand_coords) > 0:
                hand_landmarks = hand_coords[0]
                hand_landmarks_over_time.append(hand_landmarks)
            else:
                hand_landmarks_over_time.append([[0, 0]]*21)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            hand_landmarks_over_time.append([[0, 0]]*21)
    else:
        hand_landmarks_over_time.append([[0, 0]]*21)

hand_landmarks_over_time = np.array(hand_landmarks_over_time)

# 애니메이션 생성
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)  # y축 반전 (이미지 좌표계와 일치하도록)

scat = ax.scatter([], [], s=50)

def update(frame):
    landmarks = hand_landmarks_over_time[frame]
    x = [coord[0] for coord in landmarks]
    y = [coord[1] for coord in landmarks]
    scat.set_offsets(np.c_[x, y])
    ax.set_title(f'Frame {frame}')
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(hand_landmarks_over_time), interval=50, blit=True)
plt.show()

# 애니메이션을 GIF로 저장 (필요 시)
ani.save('hand_landmarks_animation.gif', writer='imagemagick')

# Bi-directional LSTM 모델 정의 및 학습
model_lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64)),
    Dense(y_train.shape[1])
])

model_lstm.compile(optimizer='adam', loss='mse')

# history 객체에 반환값 저장
history = model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss = model_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# 예측 및 시각화
y_pred = model_lstm.predict(X_test)

# 6. 모델 학습 곡선 시각화
# 모델 학습 시 반환된 history 객체를 이용하여 손실 곡선 시각화
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()





# ====================== 추가된 시각화 코드 끝 ======================