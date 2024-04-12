# Importar bibliotecas
import numpy as np
import cv2 as cv

# Definir constantes
threshold = 10

# Definir función de flujo óptico
def calculate_flow(old_frame, new_frame):
    # Reducir la resolución de la imagen
    old_frame = cv.resize(old_frame, (int(old_frame.shape[1] / 2), int(old_frame.shape[0] / 2)))
    new_frame = cv.resize(new_frame, (int(new_frame.shape[1] / 2), int(new_frame.shape[0] / 2)))

    result = cv.calcOpticalFlowFarneback(old_frame, new_frame, None, 0.5, 3, 15, 3, 7, 1.2, 0)
    u = result[:, :, 0]  # Obtener el componente u
    v = result[:, :, 1]  # Obtener el componente v

    return u, v

# Definir función de dibujo de vectores
def draw_vectors(frame, u, v):
    height, width = frame.shape[:2]

    for i in range(0, height, 10):
        for j in range(0, width, 10):
            # Solo dibujar vectores si el flujo óptico es significativo
            if i < u.shape[0] and j < u.shape[1] and np.abs(u[i, j]) > threshold and np.abs(v[i, j]) > threshold:
                cv.arrowedLine(frame, (j, i), (int(j + u[i, j]), int(i + v[i, j])), (0, 255, 0), 1)

# Iniciar la captura de video
cap = cv.VideoCapture(0)  # Para capturar desde la cámara
# cap = cv.VideoCapture("path/to/your/video.mp4")  # Para capturar desde un archivo

# Leer el primer cuadro y convertirlo a escala de grises
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

    # Calcular el flujo óptico
    u, v = calculate_flow(old_gray, frame_gray)

    # Dibujar vectores
    draw_vectors(frame, u, v)

    cv.imshow('Flow', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Actualizar el cuadro anterior
    old_gray = frame_gray.copy()

# Cerrar todas las ventanas
cv.destroyAllWindows()
