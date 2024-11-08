import cv2
import numpy as np

def weighted_blending(img1,img2,img1_weight=0.5):
    output = np.zeros(img1.shape, dtype = np.uint8)
    height, width = img1.shape[0:2]
    if len(img1.shape)==2:
        for y in range(height):
            for x in range(width):
                output[y, x]= int(img1[y,x]*img1_weight + 
                                  img2[y,x]*(1-img1_weight))
    elif len(img1.shape==3):
        for y in range(height):
            for x in range(width):
                blended = img1[y,x]*img1_weight+img2[y,x]*(1-img1_weight)
                output[y,x] = blended.astype(np.uint8)
    return output
