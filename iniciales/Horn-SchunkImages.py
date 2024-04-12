import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Horn-Schunck Optical Flow on Video')
parser.add_argument('video', type=str, help='input_video.mp4')
args = parser.parse_args()

# Intentar abrir el archivo de video
cap = cv.VideoCapture(args.video)

# Verificar si la captura se abrió correctamente
if not cap.isOpened():
    print('Error al abrir el archivo de video:', args.video)
    exit()

# Parámetros para el flujo óptico de Horn-Schunck
alpha = 1.0  # Parámetro de suavizado
iterations = 100  # Número de iteraciones

# Leer el primer cuadro y convertir a escala de grises
ret, old_frame = cap.read()
if not ret:
    print('No frames grabbed!')
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Convertir el cuadro a escala de grises
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calcular el flujo óptico de Horn-Schunck
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calcular la magnitud y la dirección del flujo
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Convertir la dirección del flujo a colores HSV
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # Asignar la dirección del flujo a la componente de matiz
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Asignar la magnitud del flujo a la componente de saturación
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Convertir la imagen HSV a BGR para visualización
    flow_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Visualizar la imagen de flujo óptico
    cv.imshow('Horn-Schunck Optical Flow', flow_rgb)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Actualizar el cuadro anterior
    old_gray = frame_gray.copy()

# Cerrar todas las ventanas
cv.destroyAllWindows()