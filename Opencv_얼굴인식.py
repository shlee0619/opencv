import cv2
import face_recognition
import numpy as np
from pydub import AudioSegment
import moviepy.editor as mp


# 비디오 파일 경로
video_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_VoodooChild.mp4'
audio_output_path = r'C:\Mtest\Mtest\이상혁zoomopencv\data\JimiHendrix_VoodooChild.mp3'


# 비디오 파일 열기 및 오디오 추출
video_clip = mp.VideoFileClip(video_path)
video_clip.audio.write_audiofile(audio_output_path, codec='mp3')
print("MP3로 오디오가 성공적으로 추출되었습니다.")


# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오의 첫 번째 프레임을 읽고 얼굴 인식
ret, frame = cap.read()
if not ret:
    print("비디오에서 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

# 얼굴 인식
face_locations = face_recognition.face_locations(frame)

# 인식된 얼굴 주위에 사각형 그리기
for face in face_locations:
    top, right, bottom, left = face
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

# 첫 번째 프레임을 표시
cv2.imshow('Face Detection', frame)
cv2.waitKey(0)

# 비디오 프레임 반복 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임에서 얼굴 인식
    face_locations = face_recognition.face_locations(frame)
    
    # 인식된 얼굴 주위에 사각형 그리기
    for face in face_locations:
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

    # 결과 프레임 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 해제 및 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()