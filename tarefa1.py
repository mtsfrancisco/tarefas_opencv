import cv2
import matplotlib.pyplot as plt

def calc_hist(img):
    cv2.imshow("Imagem", img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('Grayscale histogram')
    plt.xlabel('Bins')
    plt.ylabel('# number of pixels')
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()
    cv2.waitKey(0)

def mudar_intensidade(img):
    altura, largura = img.shape
    intensidade = input("Digite o valor: ")

    for i in range(altura):
        for j in range(largura):
            nova_intensidade = int(img[i, j]) + int(intensidade)
            if (nova_intensidade > 255):
                nova_intensidade = 255
            elif (nova_intensidade < 0):
                nova_intensidade = 0
            img[i, j] = nova_intensidade

    cv2.imshow("Imagem", img)
    cv2.waitKey(0)

def inverter_intensidade(img):
    altura, largura = img.shape

    for i in range(altura):
        for j in range(largura):
            img[i, j] = 255 - int(img[i, j])

    cv2.imshow("Imagem", img)
    cv2.waitKey(0)

img = cv2.imread("imagens/barco.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
calc_hist(gray)

#cv2.imshow("Imagem", gray)
#cv2.waitKey(0)

mudar_intensidade(gray)
inverter_intensidade(gray)