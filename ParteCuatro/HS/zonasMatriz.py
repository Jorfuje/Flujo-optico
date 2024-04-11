import cv2
import numpy as np
import os

output_folder = "imagenes_iteracionHS"
# Crear la carpeta para almacenar las imágenes por iteración si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

def sumar_cinco_a_promedios(promedio_file, grid_size):
    with open(promedio_file, 'r') as file:
        lines = file.readlines()
    
    new_promedios = []
    for line in lines:
        promedio = float(line.split(':')[-1].strip())
        new_promedio = promedio + 5.97
        new_promedios.append(new_promedio)
    
    # Ajustar el tamaño de new_promedios si es necesario
    expected_length = grid_size ** 2
    if len(new_promedios) < expected_length:
        new_promedios.extend([0] * (expected_length - len(new_promedios)))

    return new_promedios

def obtener_indices_region_interes():
    # Coordenadas de inicio y fin de la región de interés
    inicio = (5, 3)
    fin = (13, 16)

    # Calcular los índices correspondientes en base a las coordenadas
    inicio_fila = inicio[0] * 10
    fin_fila = (fin[0] + 1) * 10
    inicio_columna = inicio[1] * 10
    fin_columna = (fin[1] + 1) * 10

    return inicio_fila, fin_fila, inicio_columna, fin_columna

if __name__ == "__main__":
    video_path = "ParteCuatro/HS/output.mp4"
    capture = cv2.VideoCapture(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)  # Obtener FPS del video original
    
    # Obtener la cantidad de frames
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = "videos/HSF.mp4"
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # Establecer la misma FPS para el video de salida

    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()  # Leer el primer frame
    frame = old_frame.copy()

    scale_factor = 0.5
    threshold = 4

    promedio_file = "promedioCeldas.txt"
    grid_size = 20
    new_promedios = sumar_cinco_a_promedios(promedio_file, grid_size)

    inicio_fila, fin_fila, inicio_columna, fin_columna = obtener_indices_region_interes()

    cell_width = width // 20
    cell_height = height // 20
    
    blue_vectors_count = 0  # Contador de vectores azules pintados
    iteration_count = 0  # Contador de iteraciones
    
    while ret:  # Mientras haya frames disponibles
        u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

        # Coordenadas de los puntos de inicio de los vectores azules
        blue_start_points = []

        for y in range(0, frame.shape[0], cell_height):
            for x in range(0, frame.shape[1], cell_width):
                # Verificar si la celda actual está dentro de la región de interés
                if inicio_fila <= y < fin_fila and inicio_columna <= x < fin_columna:
                    magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)
                    i = y // 10  # Índice de fila de la celda en los nuevos promedios
                    j = x // 10  # Índice de columna de la celda en los nuevos promedios
                    promedio_celda = new_promedios[i * (frame.shape[1] // 10) + j]  # Promedio de la celda
                    
                    if magnitude >= promedio_celda:
                        color = (0, 0, 255)  # Rojo
                        blue_vectors_count += 1  # Incrementar el contador de vectores azules
                        cv2.rectangle(frame, (x, y), (x + 10, y + 10), color, -1)  # Rellenar celda en rojo
                    else:
                        color = (0, 255, 0)  # Verde
                        
                    # Registrar los puntos de inicio de los vectores azules
                    if color == (0, 0, 255):
                        blue_start_points.append((int(x), int(y)))
                        
                    cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), color, 1, tipLength=0.5)
                    
                cv2.rectangle(frame, (x, y), (x + cell_width, y + cell_height), (0, 255, 255), 1)  # Dibujar cuadrícula
        
        cv2.imshow("Flujo óptico", frame)
        output_video.write(frame)
        
        # Guardar la imagen resultante por iteración
        cv2.imwrite(os.path.join(output_folder, f"iteracion_{iteration_count}.jpg"), frame)

        # Crear una matriz de 0s
        matrix = np.zeros((grid_size, grid_size), dtype=int)


        # Marcar las celdas pintadas de rojo con 1s
        for point in blue_start_points:
            j = point[0] // cell_width
            i = point[1] // cell_height
            matrix[i, j] = 1

        # Guardar la matriz en un archivo de texto
        np.savetxt(os.path.join(output_folder, f"iteracion_{iteration_count}.txt"), matrix, fmt='%d')

        iteration_count += 1
        old_frame = frame.copy()

        ret, frame = capture.read()  # Leer el siguiente frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    print(total_frames, iteration_count)
    
    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
