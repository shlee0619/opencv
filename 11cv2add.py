import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./data/wing_wall.jpg')
img2 = cv2.imread('./data/yate.jpg')

if img1 is None or img2 is None:
    print("이미지를 제대로 불러오지 못했습니다.")
else:
    if img1.shape == img2.shape:
        img3 = img1 + img2
        img4 = cv2.add(img1, img2)
        
        imgs = {
            'img1': img1,
            'img2': img2,
            'img1+img2': img3,
            'cv2.add(img1,img2)': img4  # 키 이름 수정
        }
        
        for i, (k, v) in enumerate(imgs.items()):
            plt.subplot(2, 2, i+1)
            plt.imshow(v[:, :, ::-1])  # BGR을 RGB로 변환
            plt.title(k)
            plt.xticks([])
            plt.yticks([])
        
        plt.show()
    else:
        print("이미지의 크기나 채널 수가 다릅니다.")

print('11cv2add.py 문서 end')
