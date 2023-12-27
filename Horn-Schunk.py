import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Horn-Schunck Optical Flow')
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

# Parámetros para el flujo óptico de Horn-Schunck
alpha = 1.0  # Parámetro de suavizado
iterations = 100  # Número de iteraciones

# Calcular el flujo óptico de Horn-Schunck
flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Calcular la magnitud y la dirección del flujo
magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

# Convertir la dirección del flujo a colores HSV
hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255

# Asignar la dirección del flujo a la componente de matiz
hsv[..., 0] = angle * 180 / np.pi / 2

# Asignar la magnitud del flujo a la componente de saturación
hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

# Convertir la imagen HSV a BGR para visualización
flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# Visualizar la imagen de flujo óptico
cv.imshow('Horn-Schunck Optical Flow', flow_rgb)
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()
