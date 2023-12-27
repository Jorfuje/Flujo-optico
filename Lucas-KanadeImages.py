import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('image1', type=str, help='first_image.jpg')
parser.add_argument('image2', type=str, help='second_image.jpg')
args = parser.parse_args()

# Leer las dos imágenes
old_frame = cv.imread(args.image1)
frame = cv.imread(args.image2)

# Verificar si las imágenes se cargaron correctamente
if old_frame is None or frame is None:
    print('Error al abrir una o ambas imágenes.')
    exit()

# Convertir las imágenes a escala de grises
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Parámetros para la detección de esquinas ShiTomasi
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Encontrar esquinas en la primera imagen
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear una máscara de imagen para propósitos de dibujo
mask = np.zeros_like(old_frame)

# Calcular el flujo óptico
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

# Seleccionar puntos buenos
if p1 is not None:
    good_new = p1[st == 1]
    good_old = p0[st == 1]

# Dibujar las pistas
color = np.random.randint(0, 255, (100, 3))
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

img = cv.add(frame, mask)
cv.imshow('frame', img)
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()
