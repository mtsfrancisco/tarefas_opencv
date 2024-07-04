import cv2
import numpy as np


kernel_prewitt_horizontal = np.array([[-1, 0, 1],
                                      [-1, 0, 1], 
                                      [-1, 0, 1]])  

def rotate(kernel):
    kernel_copy = kernel.copy()

    for i in range (kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_copy[i, j] = kernel[kernel.shape[0] - 1 - i, kernel.shape[1] - 1 - j]
    return kernel_copy

def conv(image, kernel):
    kernel = rotate(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h//2
    w = kernel_w//2

    image_conv = np.zeros(image.shape)
    first_value = 0

    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-w+n]    
                    if sum > 255:
                        sum = 255
                    elif sum < 0:
                        sum = 0       
            image_conv[i][j] = sum

    #image_conv = cv2.normalize(image_conv, None, 0, 255, cv2.NORM_MINMAX)
    return image_conv.astype("uint8")

image = cv2.imread("imagens/pikachu2ComBackground.webp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
convol = conv(gray, kernel_prewitt_horizontal)
cv2.imshow("Convolucao", convol)


kernel = rotate(kernel_prewitt_horizontal)
convolucao = cv2.filter2D(gray, -1, kernel)
cv2.imshow("OpenCV", convolucao)
cv2.waitKey(0)