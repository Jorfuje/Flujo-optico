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

    # Inicializar lista para almacenar los promedios de magnitud por celda
    promedios_magnitud_por_celda = []

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
        threshold = 1.5

        # Inicializar algunas variables para el análisis de flujo óptico
        magnitudes_por_celda = []

        # Dividir la cuadrícula en una matriz de 20x20
        grid_size = 20
        grid_width = u.shape[1] // grid_size
        grid_height = u.shape[0] // grid_size

        # Iterar sobre cada celda de la cuadrícula
        for i in range(grid_size):
            for j in range(grid_size):
                # Obtener la región de interés (celda) del flujo óptico
                grid_u = u[i * grid_height: (i + 1) * grid_height, j * grid_width: (j + 1) * grid_width]
                grid_v = v[i * grid_height: (i + 1) * grid_height, j * grid_width: (j + 1) * grid_width]

                # Calcular la magnitud del vector de flujo óptico en cada celda
                magnitude = np.sqrt(grid_u ** 2 + grid_v ** 2)

                # Calcular el promedio de la magnitud de los vectores en la celda actual
                promedio_magnitud = np.mean(magnitude)
                magnitudes_por_celda.append(promedio_magnitud)

                # Escribir la información en el archivo de texto
                output_file.write(f"Para la celda ({i}, {j}) de las imágenes {image1_path} y {image2_path}:\n")
                output_file.write(f"Promedio de magnitud: {promedio_magnitud}\n\n")

        # Agregar los promedios de magnitud por celda a la lista
        promedios_magnitud_por_celda.append(magnitudes_por_celda)

    # Calcular el promedio final por celda
    promedios_finales_celda = np.mean(np.array(promedios_magnitud_por_celda), axis=0)

    # Guardar los promedios finales por celda en un archivo
    with open("promedioCeldas.txt", "w") as output_celdas_file:
        for i in range(grid_size):
            for j in range(grid_size):
                output_celdas_file.write(f"Promedio final de celda ({i}, {j}): {promedios_finales_celda[i * grid_size + j]}\n")

    # Cerrar el archivo de texto
    output_file.close()
