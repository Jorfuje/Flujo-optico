import cv2
import numpy as np
import os

window = "Lucas-Kanade Optical Flow"
video_path = "ParteCuatro/LK/ascensor/output.mp4"
output_video_path = "videos/LK.mp4"
promedios_file = "promedio_final_por_celdaLK.txt"
output_folder = "imagenes_iteracionLK"

# Crear la carpeta para almacenar las imágenes por iteración si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

capture = cv2.VideoCapture(video_path)

# Obtener la duración total del video original
fps = capture.get(cv2.CAP_PROP_FPS)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames, 'Total frame')

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

# Inicializar la lista de promedios
promedios = []

# Leer los promedios de las magnitudes por celda desde el archivo
with open(promedios_file, 'r') as file:
    for line in file:
        if line.startswith("Promedio final de celda"):
            promedio = float(line.split(":")[-1])
            promedios.append(promedio + 2.5)

# Inicializar algunas variables
old_gray = None
p0 = None
blue_vectors_count = 0  # Contador de vectores azules pintados
iteration_count = 0  # Contador de iteraciones

while True:
    # Capturar el cuadro actual del video
    ret, frame = capture.read()
    
    # Coordenadas de los puntos de inicio de los vectores azules
    blue_start_points = []

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

            # Dibujar los vectores de movimiento y detectar celdas con magnitud mayor al promedio
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a - c, b - d)

                # Obtener el índice de la celda actual
                cell_x = int(c // cell_width)
                cell_y = int(d // cell_height)
                
                # Obtener el promedio de magnitud de la celda actual
                promedio_celda = promedios[cell_y * 20 + cell_x]

                # Filtrar vectores con magnitud significativa
                magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
                if 3 <= cell_x < 15 and 5 <= cell_y < 13:
                    if magnitude > 0.5:  # Mantenemos el umbral en 0.5
                        if magnitude > promedio_celda: # Comparar con el promedio de la celda
                            color = (255, 0, 0)  # Pintar de rojo si la magnitud es mayor al promedio
                            blue_vectors_count += 1  # Incrementar el contador de vectores azules
                        else:
                            color = (0, 255, 0)  # Pintar de verde si la magnitud es menor o igual al promedio
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
                if 3 <= cell_x < 15 and 5 <= cell_y < 13:
                    cv2.rectangle(frame, (cell_x * cell_width, cell_y * cell_height), ((cell_x + 1) * cell_width, (cell_y + 1) * cell_height), (0, 0, 255), -1)

            # Actualizar los puntos anteriores
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = gray.copy()

    # Dibujar la cuadrícula en cada celda
    for i in range(0, width, cell_width):
        cv2.line(frame, (i, 0), (i, height), (0, 255, 255), 1)
    for j in range(0, height, cell_height):
        cv2.line(frame, (0, j), (width, j), (0, 255, 255), 1)

    # Guardar la imagen con la cuadrícula y los vectores en una carpeta
    output_image_path = os.path.join(output_folder, f"iteracion_{iteration_count}.jpg")
    cv2.imwrite(output_image_path, frame)

    # Matriz del video en un archivo de texto
    matrix = np.zeros((20, 20), dtype=int)
    for x, y in blue_start_points:
        cell_x = x // cell_width
        cell_y = y // cell_height
        if 3 <= cell_x < 15 and 5 <= cell_y < 13:
            matrix[cell_y, cell_x] = 1
    iteration_filename = os.path.join(output_folder, f"iteracion_{iteration_count}.txt")
    np.savetxt(iteration_filename, matrix, fmt='%d')

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
