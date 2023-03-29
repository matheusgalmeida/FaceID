import cv2
img = cv2.imread('MatheusAlmeida2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cria um objeto GPUMat a partir do array NumPy
d_gray = cv2.cuda_GpuMat()
d_gray.upload(gray)

# Aplica a operação de thresholding usando a GPU
retval, d_th = cv2.cuda.threshold(d_gray, 0, 255, cv2.THRESH_BINARY)

# Baixa o resultado de volta para o array NumPy
th = d_th.download()

# Exibe o resultado
cv2.imshow('threshold', th)
cv2.waitKey(0)
