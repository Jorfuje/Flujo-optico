import cv2
import numpy as np

window = "Lucas-Kanade Optical Flow"
video_path = "video.mp4"
output_video_path = "videos/LK.mp4"

capture = cv2.VideoCapture(video_path)

# Obtener el tamaño del video de entrada
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear un objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')

output_video = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height))

# Parámetros para el algoritmo de Lucas-Kanade
lk_params = dict(winSize=(50, 50), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

# Inicializar algunas variables
old_gray = None
p0 = None

while True:
    # Capturar el cuadro actual del video
    ret, frame = capture.read()

    # Si no hay datos, salir del bucle
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Si es el primer cuadro o después de cierto tiempo, detectar esquinas nuevamente
    if old_gray is None or cv2.getTickCount() % 50 == 0:
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7, k=0.04)
        old_gray = gray.copy()

    else:
        # Calcular el flujo óptico con el algoritmo de Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        # Verificar que el cálculo del flujo óptico fue exitoso
        if p1 is not None:
            # Seleccionar puntos buenos
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Dibujar los vectores de movimiento
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a - c, b - d)
                cv2.arrowedLine(frame, (int(c), int(d)), (int(c + 5*vector[0]), int(d + 5*vector[1])), (0, 255, 0), 2)

            # Actualizar los puntos anteriores
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = gray.copy()

    # Guardar el cuadro en el video de salida
    output_video.write(frame)

    # Mostrar la imagen
    cv2.imshow(window, frame)

    # Esperar por 30 ms a que se presione una tecla
    key = cv2.waitKey(30) & 0xFF

    # Si la tecla es ESC, salir
    if key == 27:
        break

# Liberar los recursos y cerrar la ventana
capture.release()
output_video.release()
cv2.destroyAllWindows()

