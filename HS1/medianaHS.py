import cv2
import numpy as np
import os

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Calcular el flujo óptico con el algoritmo de Horn-Schunck
    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    # Extraer las componentes horizontal y vertical del flujo óptico
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

if __name__ == "__main__":
    # Obtener la lista de archivos en la carpeta de imágenes
    imagenes_dir = "imagenes1"
    imagenes_files = sorted(os.listdir(imagenes_dir))
    imagenes_paths = [os.path.join(imagenes_dir, filename) for filename in imagenes_files]

    # Crear o abrir un archivo de texto para guardar la información
    output_file = open("informacion_flujo_optico.txt", "w")

    # Inicializar listas para almacenar las magnitudes de cada par de imágenes
    magnitudes_por_par = []

    # Iterar sobre cada par de imágenes consecutivas
    for i in range(len(imagenes_paths) - 1):
        image1_path = imagenes_paths[i]
        image2_path = imagenes_paths[i + 1]

        # Cargar las dos imágenes
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # Calcular el flujo óptico entre los fotogramas consecutivos
        u, v = horn_schunck_real_time(image1, image2, 0.01, 1)  # Cambiar alpha a 0.01 para reducir el ruido

        # Ajustar los parámetros para obtener una visualización más clara
        scale_factor = .8
        threshold = 3.5

        # Inicializar algunas variables para el análisis de flujo óptico
        total_vectors = 0
        magnitudes = []

        # Iterar sobre cada píxel del fotograma
        for y in range(0, image1.shape[0], 10):
            for x in range(0, image1.shape[1], 10):
                # Calcular la magnitud del vector de flujo óptico en el píxel actual
                magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)

                # Si la magnitud supera el umbral, contar el vector y almacenar su magnitud
                if magnitude > threshold:
                    total_vectors += 1
                    magnitudes.append(magnitude)

                    # Guardar la magnitud del vector en el archivo de texto
                    output_file.write(f"Magnitud del vector: {magnitude}\n")

        # Calcular la mediana de las magnitudes de los vectores significativos
        if magnitudes:
            median_magnitude = np.median(magnitudes)
            magnitudes_por_par.append(median_magnitude)

        # Si hay vectores pintados, escribir la información en el archivo de texto
        if total_vectors > 0:
            output_file.write(f"Para las imágenes {image1_path} y {image2_path}:\n")
            output_file.write(f"Total de vectores pintados: {total_vectors}\n")
            output_file.write(f"Mediana del tamaño de los vectores: {median_magnitude}\n\n")

    # Cerrar el archivo de texto
    output_file.close()

    # Calcular la mediana final de las medianas de cada par de imágenes
    if magnitudes_por_par:
        mediana_final_de_medianas = np.median(magnitudes_por_par)
    else:
        mediana_final_de_medianas = 0

    # Imprimir resultados
    print("Mediana final de medianas:", mediana_final_de_medianas)
