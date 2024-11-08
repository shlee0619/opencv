import cv2
import numpy as np
from matplotlib import pyplot as plt


img_gray = cv2.imread('./data/Paris.png', 0)
_, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
adt_mean = cv2.adaptiveThreshold(img_gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, 2)
adt_gaus = cv2.adaptiveThreshold(img_gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 2)

titles = ['Gray', 'Global', 'Mean', 'Gaussian']
images = [img_gray, img_bin, adt_mean, adt_gaus]

plt.figure(figsize=(8,6))
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()