import cv2
import numpy as np

# Função para calcular o threshold adaptativo manualmente
def adaptive_threshold_mean_c(image, block_size, C):
    altura, largura = image.shape
    half_block = block_size // 2
    result = np.zeros((altura, largura), dtype=np.uint8)
    
    for y in range(altura):
        for x in range(largura):
            # Defina as bordas do bloco
            y1 = max(0, y - half_block)
            y2 = min(altura, y + half_block + 1)
            x1 = max(0, x - half_block)
            x2 = min(largura, x + half_block + 1)
            
            # Calcule a média dos pixels no bloco
            bloco = image[y1:y2, x1:x2]
            media = np.mean(bloco)
            
            # Calcule o threshold
            threshold = media - C
            
            # Aplique o threshold
            if image[y, x] > threshold:
                result[y, x] = 255
            else:
                result[y, x] = 0
    
    return result

# Passo 1: Abra a imagem em escala de cinza
imagem = cv2.imread('imagens/pikachu2ComBackground.webp', cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao abrir a imagem.")
    exit()

# Parâmetros para o thresholding adaptativo
block_size = 11  # Tamanho do bloco (deve ser ímpar e >= 3)
C = 2  # Constante subtraída da média

# Passo 2: Aplique o thresholding adaptativo manualmente
imagem_thresh = adaptive_threshold_mean_c(imagem, block_size, C)

# Passo 3: Salve ou mostre a imagem modificada
cv2.imshow('Imagem com Threshold Adaptativo', imagem_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()