import cv2
import numpy as np

window = "Lucas-Kanade Optical Flow"
video_path = "SegundaParte/video.mp4"
output_video_path = "videos/LK.mp4"

capture = cv2.VideoCapture(video_path)

# Obtener la duración total del video original
fps = capture.get(cv2.CAP_PROP_FPS)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Obtener el tamaño del video de entrada
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calcular el tamaño de cada cuadro en la matriz 20x20
cell_width = width // 20
cell_height = height // 20

# Crear un objeto VideoWriter para el video resultante
fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=200, qualityLevel=0.0008, minDistance=10, blockSize=5, k=0.004)
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
            suma=0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a - c, b - d)
                magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
                if magnitude > 0.5:  # Solo dibujar si la magnitud supera el umbral
                    color = (0, 255, 0) if magnitude <= 5 else (0, 0, 255) if magnitude <= 7 else (255, 0, 0)
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(c + 5 * vector[0]), int(d + 5 * vector[1])), color, 2)

                                # Contar los vectores que superan el umbral
                    suma += 1
                    print(suma)

                # Pintar la celda de rojo si hay 13 o más vectores que superan el umbral
            if suma >= 13:
                for old in good_old:
                    c, d = old.ravel()
                    cell_x = c // cell_width
                    cell_y = d // cell_height
                    cv2.rectangle(frame, (int(cell_x * cell_width), int(cell_y * cell_height)), (int((cell_x + 1) * cell_width), int((cell_y + 1) * cell_height)), (0, 0, 255), -1)


            # Actualizar los puntos anteriores
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = gray.copy()

    # Escribir el cuadro en el video de salida
    output_video.write(frame)

    # Mostrar la imagen
    cv2.imshow(window, frame)

    # Esperar el tiempo necesario para mantener la velocidad del video original
    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    # Si la tecla es ESC, salir
    if key == 27:
        break

# Liberar los recursos y cerrar la ventana
capture.release()
output_video.release()
cv2.destroyAllWindows()

