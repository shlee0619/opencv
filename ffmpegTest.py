import os

# FFmpeg 경로 확인
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'

# FFmpeg가 작동하는지 확인
if os.path.exists(ffmpeg_path):
    print("FFmpeg가 정상적으로 설치되어 있습니다.")
    os.system(ffmpeg_path + ' -version')
else:
    print("FFmpeg를 찾을 수 없습니다. 경로를 확인하세요.")