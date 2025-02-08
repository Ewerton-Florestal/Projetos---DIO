
import cv2
import numpy as np

img = cv2.imread("Cachorro.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

suave = cv2.GaussianBlur(img, (7,7),0) # aplica blur

(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)

(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
resultado = np.vstack([
    np.hstack([img, suave]),
    np.hstack([bin, binI])])

cv2.imshow("Binarização da Imagem", resultado)
cv2.waitKey(0)

