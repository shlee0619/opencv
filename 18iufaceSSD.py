import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import cv2  
# pip install opencv-python
#-----------------------------------------------------------------------------------
print('🍀 ' * 30)
print()
# MobileNet과 SSD(Single Shot MultiBox Detector)를 활용한 효율적인 객체 인식
print( 'MobileNet과 SSD(Single Shot MultiBox Detector)')

def img_show(title='Single Shot MultiBox Detector', img=None, figsize=(10,8)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        

prototxt_path = './data/MobileNetSSD_deploy.prototxt.txt'
model_path = './data/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

kind = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
 
LABEL_COLORS = np.random.uniform(0, 255, size=(len(kind), 3))
img = cv2.imread("./data/example_05.jpg")

(h, w) = img.shape[:2]
resized = cv2.resize(img, (500, 500))
blob = cv2.dnn.blobFromImage(resized, 0.007843, (500, 500), 127.5)
 
net.setInput(blob)
detections = net.forward() #중요

myCopy = img.copy()
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
 
    if confidence > 0.2 :
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        print("객체정보 {} : [ {:.2f} % ]".format(kind[idx], confidence * 100))
        
        cv2.rectangle(myCopy, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(myCopy, "{} : {:.2f}%".format(kind[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLORS[idx], 2)
        

img_show('SSD(Single Shot MultiBox Detector)기술 ', myCopy, figsize=(10,8))

#참고사이트 