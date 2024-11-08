import cv2
import numpy as np

img1 = cv2.imread('./data/logo.png')
img1_resized = cv2.resize(img1, (225, 225))

img2 = cv2.imread('./data/lena.jpg')



rows, cols, channels = img1_resized.shape

mask = np.full_like(img1_resized, 255)
height, width = img2.shape[:2]
center = (width//2, height//2)

normal = cv2.seamlessClone(img1_resized, img2, mask, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(img1_resized, img2, mask, center, cv2.MIXED_CLONE)


cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey(0)
cv2.destroyAllWindows()