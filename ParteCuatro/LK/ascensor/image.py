import cv2
import numpy as np
import os

input_image_folder = "imagenes1"
output_image_folder = "imagenes_salida"
output_file_path = "promedio_por_celda_por_frameLK.txt"
final_output_file_path = "promedio_final_por_celdaLK.txt"

# Crear la carpeta de salida si no existe
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)

# Obtener la lista de nombres de archivos de imágenes en la carpeta de entrada
image_files = sorted(os.listdir(input_image_folder))

# Leer la primera imagen para obtener su tamaño
first_image = cv2.imread(os.path.join(input_image_folder, image_files[0]))
height, width, _ = first_image.shape

# Calcular el tamaño de cada cuadro en la matriz 20x20
cell_width = width // 20
cell_height = height // 20

# Inicializar la matriz de salida
output_matrix = np.zeros((20, 20), dtype=int)

# Parámetros para el algoritmo de Lucas-Kanade
lk_params = dict(winSize=(50, 50), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

# Inicializar algunas variables
old_gray = None
p0 = None
iteration_count = 0  # Contador de iteraciones
cell_magnitude_sum = np.zeros((20, 20))  # Matriz para almacenar la suma de magnitudes por celda
cell_count = np.zeros((20, 20), dtype=int)  # Matriz para almacenar el conteo de vectores por celda

# Iterar sobre los nombres de archivos de imágenes
for image_file in image_files:
    # Leer la imagen actual
    frame = cv2.imread(os.path.join(input_image_folder, image_file))

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Si es el primer cuadro o después de cierto tiempo, detectar esquinas nuevamente
    if old_gray is None or iteration_count % 50 == 0:
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

            # Dibujar los vectores de movimiento y detectar celdas con magnitud mayor al promedio
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                vector = (a - c, b - d)

                # Obtener el índice de la celda actual
                cell_x = int(c // cell_width)
                cell_y = int(d // cell_height)
                
                

                # Filtrar vectores con magnitud significativa
                magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
                if 3 <= cell_x < 15 and 5 <= cell_y < 13:
                    if magnitude > 0.5:  # Mantenemos el umbral en 0.5
                        if magnitude > 3.5: # Comparar con el promedio de la celda
                            color = (255, 0, 0)  # Pintar de rojo si la magnitud es mayor al promedio
                            cv2.rectangle(frame, (cell_x * cell_width, cell_y * cell_height), ((cell_x + 1) * cell_width, (cell_y + 1) * cell_height), color, -1)  # Pintar la celda de rojo
                            blue_start_points.append((int(c), int(d)))  # Registrar los puntos de inicio de los vectores azules
                        else:
                            color = (0, 255, 0)  # Pintar de verde si la magnitud es menor o igual al promedio
                        cv2.arrowedLine(frame, (int(c), int(d)), (int(c + 5 * vector[0]), int(d + 5 * vector[1])), color, 2)

                    else:
                        # Remover los puntos de inicio de los vectores que ya no están presentes
                        blue_start_points = [(x, y) for x, y in blue_start_points if any((abs(x - px) > 5 or abs(y - py) > 5) for px, py in good_new)]

                # Pintar la celda donde comienza un vector azul de color rojo
                for x, y in blue_start_points:
                    cell_x = x // cell_width
                    cell_y = y // cell_height
                    cell_count[cell_y, cell_x] += 1  # Incrementar el conteo de vectores por celda
                    output_matrix[cell_y, cell_x] = 1  # Establecer el valor en la matriz de salida como 1

            # Actualizar los puntos anteriores
            p0 = good_new.reshape(-1, 1, 2)
            old_gray = gray.copy()

    # Guardar la imagen con los vectores dibujados
    output_frame = frame.copy()
    for i in range(0, width, cell_width):
        cv2.line(output_frame, (i, 0), (i, height), (0, 255, 255), 1)
    for j in range(0, height, cell_height):
        cv2.line(output_frame, (0, j), (width, j), (0, 255, 255), 1)

    # Guardar la imagen resultante
    output_image_path = os.path.join(output_image_folder, f"output_{iteration_count}.jpg")
    cv2.imwrite(output_image_path, output_frame)

    # Incrementar el contador de iteraciones
    iteration_count += 1

# Liberar los recursos y cerrar la ventana
cv2.destroyAllWindows()
