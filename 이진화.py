import cv2
import numpy as np

def two_tone(img, threshold = 128):
    output = (img>threshold) * 255
    return output.astype(np.uint8)

img = cv2.imread('./data/lena.jpg', 0)
new_img = two_tone(img, threshold=120)
cv2.imwrite('./data/lena_bin.jpg', new_img)
cv2.imshow('img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()