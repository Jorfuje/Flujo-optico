import cv2
import numpy as np

# Cargar las dos imágenes
image1 = cv2.imread("SegundaParte/image1.jpg")
image2 = cv2.imread("SegundaParte/image2.jpg")

# Obtener el tamaño de las imágenes
height, width, _ = image1.shape

# Parámetros para el algoritmo de Lucas-Kanade
lk_params = dict(winSize=(50, 50), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

# Convertir las imágenes a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detectar características en la primera imagen
p0 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=200, qualityLevel=0.0008, minDistance=10, blockSize=5, k=0.004)

# Inicializar algunas variables
old_gray = gray1.copy()

# Calcular el flujo óptico con el algoritmo de Lucas-Kanade
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray2, p0, None, **lk_params)

# Seleccionar puntos buenos
good_new = p1[st == 1]
good_old = p0[st == 1]

# Crear una copia de la segunda imagen
result_image = image2.copy()

# Dibujar los vectores de movimiento en la imagen resultante
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    vector = (a - c, b - d)

    # Filtrar vectores con magnitud significativa
    magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if magnitude > 0.5:  # Ajusta el umbral según sea necesario
        cv2.arrowedLine(result_image, (int(c), int(d)), (int(c + 5 * vector[0]), int(d + 5 * vector[1])), (0, 255, 0), 2)

# Guardar la imagen resultante
cv2.imwrite("output_imageLK.jpg", result_image)
print("Imagen guardada como:", "output_imageLK.jpg")

# Mostrar la imagen
cv2.imshow("Resultado con vectores", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
