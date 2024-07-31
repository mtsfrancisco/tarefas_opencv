import numpy as np
import cv2

def adicao(img, img2):
    altura, largura = img.shape
    for i in range(altura):
        for j in range(largura):
            soma = int(img[i, j]) + int(img2[i, j])
            if soma > 255:
                soma = 255
            elif soma < 0:
                soma = 0
            img[i, j] = soma
    return img

def subtracao(img, img2):
    altura, largura = img.shape
    for i in range(altura):
        for j in range(largura):
            sub = int(img[i, j]) - int(img2[i, j])
            if sub < 0:
                sub = 0
            img[i, j] = sub
    return img

img = cv2.imread('imagens/pikachu2ComBackground.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_copy = gray.copy()

img2 = np.ones(gray.shape, np.uint8) * 100
img2_copy = img2.copy()

img3 = adicao(gray, img2)

img4= subtracao(gray_copy, img2_copy)


cv2.imshow('Subtracao', img4)
cv2.imshow('Adicao', img3)
cv2.waitKey(0)
    