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
blue_vectors_count = 0  # Contador de vectores azules pintados
iteration_count = 0  # Contador de iteraciones

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

            # Coordenadas de los puntos de inicio de los vectores azules
            blue_start_points = []

            # Dibujar los vectores de movimiento y detectar celdas con magnitud mayor a 1.5
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a - c, b - d)

                # Filtrar vectores con magnitud significativa
                magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
                if magnitude > 0.5:  # Mantenemos el umbral en 0.5
                    if magnitude > 6:
                        color = (255, 0, 0)  # Pintar de azul si la magnitud es mayor a 1.5
                        blue_vectors_count += 1  # Incrementar el contador de vectores azules
                    else:
                        color = (0, 255, 0)  # Pintar de verde si la magnitud es menor o igual a 1.5
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(c + 5 * vector[0]), int(d + 5 * vector[1])), color, 2)

                    # Registrar los puntos de inicio de los vectores azules
                    if color == (255, 0, 0):
                        blue_start_points.append((int(c), int(d)))
                else:
                    # Remover los puntos de inicio de los vectores que ya no están presentes
                    blue_start_points = [(x, y) for x, y in blue_start_points if any((abs(x - px) > 5 or abs(y - py) > 5) for px, py in good_new)]

            # Pintar la celda donde comienza un vector azul de color rojo
            for x, y in blue_start_points:
                cell_x = x // cell_width
                cell_y = y // cell_height
                cv2.rectangle(frame, (cell_x * cell_width, cell_y * cell_height), ((cell_x + 1) * cell_width, (cell_y + 1) * cell_height), (0, 0, 255), -1)

            # Actualizar los puntos anteriores
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = gray.copy()

    # Dibujar la cuadrícula en cada celda
    for i in range(0, width, cell_width):
        cv2.line(frame, (i, 0), (i, height), (0, 255, 255), 1)
    for j in range(0, height, cell_height):
        cv2.line(frame, (0, j), (width, j), (0, 255, 255), 1)

    # Escribir el cuadro en el video de salida
    output_video.write(frame)

    # Mostrar la imagen con la cuadrícula
    cv2.imshow(window, frame)

    # Imprimir el número de iteración y el número de vectores azules pintados en esta iteración
    print("Iteración:", iteration_count, "- Número de vectores azules pintados:", blue_vectors_count)

    # Reiniciar el contador de vectores azules para la próxima iteración
    blue_vectors_count = 0

    # Incrementar el contador de iteraciones
    iteration_count += 1

    # Esperar el tiempo necesario para mantener la velocidad del video original
    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    # Si la tecla es ESC, salir
    if key == 27:
        break

# Liberar los recursos y cerrar la ventana
capture.release()
output_video.release()
cv2.destroyAllWindows()
