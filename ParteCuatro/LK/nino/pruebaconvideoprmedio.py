import cv2
import numpy as np

window = "Lucas-Kanade Optical Flow"
video_path = "ParteCuatro/LK/nino/nino.mp4"
output_video_path = "videos/LK.mp4"
output_file_path = "promedio_por_celda_por_frameLKnino.txt"
final_output_file_path = "promedio_final_por_celdaLKnino.txt"

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
cell_magnitude_sum = np.zeros((20, 20))  # Matriz para almacenar la suma de magnitudes por celda
cell_count = np.zeros((20, 20), dtype=int)  # Matriz para almacenar el conteo de vectores por celda

# Lista para almacenar los promedios por celda por cada par de frames
cell_averages_per_frame = []

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
                    if magnitude > 3.5: # caida
                    #if magnitude > 2: # fractura
                        color = (255, 0, 0)  # Pintar de azul si la magnitud es mayor a 1.5
                        blue_vectors_count += 1  # Incrementar el contador de vectores azules
                    else:
                        color = (0, 255, 0)  # Pintar de verde si la magnitud es menor o igual a 1.5
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(c + 5 * vector[0]), int(d + 5 * vector[1])), color, 2)

                    # Registrar los puntos de inicio de los vectores azules
                    if color == (255, 0, 0):
                        blue_start_points.append((int(c), int(d)))
                        
                    # Calcular las coordenadas de la celda correspondiente
                    cell_x = min(int(c // cell_width), 19)
                    cell_y = min(int(d // cell_height), 19)
                    
                    # Actualizar la suma de magnitudes y el conteo de vectores por celda
                    cell_magnitude_sum[cell_y, cell_x] += magnitude
                    cell_count[cell_y, cell_x] += 1
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

    # Calcular el promedio por celda de la magnitud de los vectores
    average_magnitude_per_cell = cell_magnitude_sum / np.maximum(cell_count, 1)

    # Guardar el promedio por celda de la magnitud de los vectores en el archivo
    with open(output_file_path, 'a') as file:
        file.write("Para cada par de frames:\n")
        for i in range(len(cell_averages_per_frame) - 1):
            file.write("Para la celda (x, y) de las imágenes image_{:04d}.jpg y image_{:04d}.jpg:\n".format(i, i + 1))
            for y in range(20):
                for x in range(20):
                    file.write("Para la celda ({}, {}) de las imágenes image_{:04d}.jpg y image_{:04d}.jpg:\n".format(x, y, i, i + 1))
                    file.write("Promedio de magnitud: {}\n".format(cell_averages_per_frame[i][y, x]))
            file.write("\n")


    # Añadir los promedios por celda a la lista
    cell_averages_per_frame.append(average_magnitude_per_cell)

# Calcular el promedio final de los promedios por celdas
final_average_per_cell = np.mean(cell_averages_per_frame, axis=0)

# Guardar el promedio final de los promedios por celdas en el archivo final
with open(final_output_file_path, 'w') as final_file:
    final_file.write("Promedio final por celda:\n")
    for y in range(20):
        for x in range(20):
            final_file.write("Promedio final de celda ({}, {}): {}\n".format(x, y, final_average_per_cell[y, x]))

# Liberar los recursos y cerrar la ventana
capture.release()
output_video.release()
cv2.destroyAllWindows()

