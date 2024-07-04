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
    print(kernel_copy)
    print("=======KERNEL=======")
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

    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-w+n]        
            image_conv[i][j] = sum

    #image_conv = cv2.normalize(image_conv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_conv.astype(np.uint8)

image = cv2.imread("imagens/pikachu2ComBackground.webp")
img = image[0:5, 0:5]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray)
print("====================")
convol = conv(gray, kernel_prewitt_horizontal)
print(convol)


