import cv2
import numpy as np

def ez_blending(img1, img2, img1_weight=0.5):
    output = img1*img1_weight+img2*(1-img1_weight)
    return output.astype(np.uint8)

new_img = ez_blending(img1, img2, img1_weight=0.3)
cv2.imshow('img', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()