import cv2
import numpy as np
import os

# Obtener la lista de archivos en la carpeta "imagenes"
imagenes_dir = "imagenes1"
imagenes_files = sorted(os.listdir(imagenes_dir))
imagenes_paths = [os.path.join(imagenes_dir, filename) for filename in imagenes_files]

# Crear o abrir un archivo de texto para guardar la información
output_file = open("informacion_flujo_optico.txt", "w")

# Inicializar contador de resultados guardados
total_resultados_guardados = 0

# Iterar sobre cada par de imágenes consecutivas
for i in range(len(imagenes_paths) - 1):
    image1_path = imagenes_paths[i]
    image2_path = imagenes_paths[i + 1]

    # Cargar las dos imágenes
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Obtener el tamaño de las imágenes
    height, width, _ = image1.shape

    # Parámetros para el algoritmo de Lucas-Kanade
    lk_params = dict(winSize=(50, 50), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

    # Convertir las imágenes a escala de grises
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detectar características en la primera imagen
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=200, qualityLevel=0.0008, minDistance=10, blockSize=5, k=0.004)

    # Inicializar algunas variables
    old_gray = gray1.copy()
    total_vectors = 0
    total_magnitude = 0

    # Calcular el flujo óptico con el algoritmo de Lucas-Kanade
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray2, p0, None, **lk_params)

    # Seleccionar puntos buenos
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Dibujar los vectores de movimiento en la imagen resultante
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        vector = (a - c, b - d)

        # Filtrar vectores con magnitud significativa
        magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if magnitude > 0.5:  # Ajusta el umbral según sea necesario
            total_vectors += 1
            total_magnitude += magnitude

    # Calcular el promedio del tamaño de los vectores
    if total_vectors > 0:
        average_magnitude = total_magnitude / total_vectors
    else:
        average_magnitude = 0

    # Si hay vectores pintados, escribir la información en el archivo de texto
    if total_vectors > 0:
        output_file.write(f"Para las imágenes {image1_path} y {image2_path}:\n")
        output_file.write(f"Total de vectores pintados: {total_vectors}\n")
        output_file.write(f"Promedio del tamaño de los vectores: {average_magnitude}\n\n")
        total_resultados_guardados += 1

# Cerrar el archivo de texto
output_file.close()

# Imprimir el total de resultados guardados
print("Total de resultados guardados:", total_resultados_guardados)

