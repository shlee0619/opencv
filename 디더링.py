import cv2
import numpy as np

def minmax(pixel):
    if pixel > 255:
        pixel = 255
    if pixel < 0:
        pixel = 0
    return pixel

def dithering_gray(inMat, samplingF):
    h = inMat.shape[0]
    w = inMat.shape[1]

    for y in range(0, h-1):
        for x in range(1, w-1):
            # threshold the pixel
            old_p = inMat[y, x]
            new_p = np.round(samplingF * old_p/255.0) * (255/samplingF)
            inMat[y, x] = new_p

            quant_error_p = old_p - new_p

            inMat[y, x+1] = minmax(inMat[y, x+1] + quant_error_p * 7/16.0)
            inMat[y+1, x-1] = minmax(inMat[y+1, x-1] +
                                      quant_error_p * 3/16.0)
            inMat[y+1, x] = minmax(inMat[y+1, x] +
                                      quant_error_p * 5/16.0)
            inMat[y+1, x+1] = minmax(inMat[y+1, x+1] +
                                      quant_error_p * 1/16.0)
    return inMat


lena = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)
lena_gray_dithering = dithering_gray(lena.copy(), 1)
cv2.imwrite('./data/lena_gray_dithering.jpg', lena_gray_dithering)
cv2.imshow('Lena Grayscale', lena)
cv2.imshow('Lena dithering', lena_gray_dithering)
cv2.waitKey(0)
cv2.destroyAllWindows()