import cv2 
import numpy as np

def adaptive_threshold_mean_c(image, block_size, C):
    altura, largura = image.shape
    half_block = block_size // 2
    result = np.zeros((altura, largura), dtype=np.uint8)
    
    for y in range(altura):
        for x in range(largura):

            y1 = max(0, y - half_block)
            y2 = min(altura, y + half_block + 1)
            x1 = max(0, x - half_block)
            x2 = min(largura, x + half_block + 1)
            

            bloco = image[y1:y2, x1:x2]
            media = np.mean(bloco)
      
            threshold = media - C
      
            if image[y, x] > threshold:
                result[y, x] = 255
            else:
                result[y, x] = 0
    
    return result

def threshhold(img, thresh):
    altura, largura = img.shape
    for i in range(altura):
        for j in range(largura):
            if img[i,j] > thresh:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

img = cv2.imread("imagens/pikachu2ComBackground.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_copy = gray.copy()
thresh_img = threshhold(gray, 100)

thresh_mean = adaptive_threshold_mean_c(gray_copy,11, 7)

cv2.imshow("Threshhold normal", thresh_img)
cv2.imshow("Mean threshhold", thresh_mean)
cv2.waitKey(0)
