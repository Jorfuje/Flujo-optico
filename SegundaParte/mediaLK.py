import cv2
import numpy as np
import os

# Obtener la lista de archivos en la carpeta "imagenes"
imagenes_dir = "imagenes1"
imagenes_files = sorted(os.listdir(imagenes_dir))
imagenes_paths = [os.path.join(imagenes_dir, filename) for filename in imagenes_files]

# Crear o abrir un archivo de texto para guardar la información
output_file = open("informacion_flujo_optico.txt", "w")

# Inicializar listas para almacenar las magnitudes de cada par de imágenes
magnitudes_por_par = []

# Inicializar contadores para el promedio total
total_vectores_totales = 0
total_magnitud_totales = 0
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
    magnitudes = []

    # Calcular el flujo óptico con el algoritmo de Lucas-Kanade
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray2, p0, None, **lk_params)

    # Seleccionar puntos buenos
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Filtrar vectores con magnitud significativa
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        vector = (a - c, b - d)

        magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if magnitude > 0.5:  # Ajusta el umbral según sea necesario
            total_vectors += 1
            total_magnitude += magnitude
            magnitudes.append(magnitude)

            # Guardar la magnitud del vector en el archivo de texto
            output_file.write(f"Magnitud del vector {i+1}: {magnitude}\n")

    # Calcular la mediana de las magnitudes de los vectores significativos
    if magnitudes:
        median_magnitude = np.median(magnitudes)
        magnitudes_por_par.append(median_magnitude)

    # Si hay vectores pintados, escribir la información en el archivo de texto
    if total_vectors > 0:
        total_vectores_totales += total_vectors
        total_magnitud_totales += total_magnitude
        total_resultados_guardados += 1

        output_file.write(f"Para las imágenes {image1_path} y {image2_path}:\n")
        output_file.write(f"Total de vectores pintados: {total_vectors}\n")
        output_file.write(f"Mediana del tamaño de los vectores: {median_magnitude}\n\n")

# Calcular promedio total de vectores y promedio total de magnitud
if total_resultados_guardados > 0:
    promedio_total_vectores = total_vectores_totales / total_resultados_guardados
    promedio_total_magnitud = total_magnitud_totales / total_resultados_guardados
else:
    promedio_total_vectores = 0
    promedio_total_magnitud = 0

# Calcular la mediana final de las medianas de cada par de imágenes
if magnitudes_por_par:
    mediana_final_de_medianas = np.median(magnitudes_por_par)
else:
    mediana_final_de_medianas = 0

# Cerrar el archivo de texto
output_file.close()

# Imprimir resultados
print("Total de resultados guardados:", total_resultados_guardados)
print("Promedio total de vectores:", promedio_total_vectores)
print("Promedio total de magnitud:", promedio_total_magnitud)
print("Mediana final de medianas:", mediana_final_de_medianas)
