import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel_prewitt_horizontal = np.array([[-1, 0, 1],
                                      [-1, 0, 1], 
                                      [-1, 0, 1]])  

gaussian = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])


def rotate(kernel):
    kernel_copy = kernel.copy()

    for i in range (kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_copy[i, j] = kernel[kernel.shape[0] - 1 - i, kernel.shape[1] - 1 - j]

    return kernel_copy

def conv(img, kernel):
    kernel = rotate(kernel)
    altura, largura = img.shape
    altura_kernel, largura_kernel = kernel.shape
    
    pad = largura_kernel // 2
    image_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output = np.zeros((altura, largura), dtype="float32")

    for y in range(altura):
        for x in range(largura):
     
            region = image_padded[y:y + altura_kernel, x:x + largura_kernel]
            k = (region * kernel).sum()
            
            
            output[y, x] = k
            
    
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    return output.astype("uint8")



img = cv2.imread("imagens/pikachu2ComBackground.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
convol = conv(gray, kernel_prewitt_horizontal)
cv2.imshow("Imagem", convol)
cv2.waitKey(0)

convolucao = cv2.filter2D(gray, -1, kernel_prewitt_horizontal)
#convolucao = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("Imagem", convolucao)
cv2.waitKey(0)

#convolucao_horizontal = convolucao(gray, kernel_prewitt_horizontal)
#cv2.imshow("Imagem", convolucao_horizontal)
#cv2.waitKey(0)


