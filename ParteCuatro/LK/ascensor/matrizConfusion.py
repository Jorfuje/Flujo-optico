import numpy as np
import os

# Función para cargar una matriz desde un archivo de texto
def load_matrix_from_file(file_path):
    return np.loadtxt(file_path, dtype=int)

# Función para calcular la matriz de confusión
def calculate_confusion_matrix(true_matrix, predicted_matrix):
    # Inicializar la matriz de confusión
    confusion_matrix = np.zeros((2, 2), dtype=int)

    # Comparar las matrices verdaderas y predichas
    for i in range(true_matrix.shape[0]):
        for j in range(true_matrix.shape[1]):
            if true_matrix[i, j] == 1:
                if predicted_matrix[i, j] == 1:
                    confusion_matrix[0, 0] += 1  # Verdadero positivo
                else:
                    confusion_matrix[0, 1] += 1  # Falso negativo
            else:
                if predicted_matrix[i, j] == 1:
                    confusion_matrix[1, 0] += 1  # Falso positivo
                else:
                    confusion_matrix[1, 1] += 1  # Verdadero negativo

    return confusion_matrix

# Carpeta que contiene los archivos de matrices verdaderas
true_matrices_folder = "imagenes_iteracionLK"
# Carpeta que contiene los archivos de matrices predichas
predicted_matrices_folder = "matrices_predichas"

# Obtener la lista de archivos en ambas carpetas
true_matrices_files = os.listdir(true_matrices_folder)
predicted_matrices_files = os.listdir(predicted_matrices_folder)

# Verificar que ambas carpetas tengan la misma cantidad de archivos
if len(true_matrices_files) != len(predicted_matrices_files):
    print("La cantidad de archivos en las carpetas no coincide. Verifica los archivos y vuelve a intentarlo.")
    exit()

# Crear el directorio para almacenar las matrices de confusión si no existe
confusion_matrices_folder = "confusion_matrices"
if not os.path.exists(confusion_matrices_folder):
    os.makedirs(confusion_matrices_folder)

# Inicializar la matriz promedio de confusión
average_confusion_matrix = np.zeros((2, 2), dtype=float)

# Procesar cada par de archivos
for true_file_name, predicted_file_name in zip(true_matrices_files, predicted_matrices_files):
    # Construir las rutas completas de los archivos
    true_file_path = os.path.join(true_matrices_folder, true_file_name)
    predicted_file_path = os.path.join(predicted_matrices_folder, predicted_file_name)

    # Cargar las matrices verdadera y predicha desde los archivos
    true_matrix = load_matrix_from_file(true_file_path)
    predicted_matrix = load_matrix_from_file(predicted_file_path)

    # Calcular la matriz de confusión
    confusion_matrix = calculate_confusion_matrix(true_matrix, predicted_matrix)

    # Imprimir la matriz de confusión para esta iteración
    print(f"Matriz de Confusión para {true_file_name} vs {predicted_file_name}:")
    print(confusion_matrix)
    print()

    # Guardar la matriz de confusión en un archivo de texto
    confusion_matrix_file_path = os.path.join(confusion_matrices_folder, f"confusion_matrix_{true_file_name[:-4]}.txt")
    np.savetxt(confusion_matrix_file_path, confusion_matrix, fmt='%d')

    # Actualizar la matriz promedio de confusión
    average_confusion_matrix += confusion_matrix

# Calcular la matriz promedio dividiendo la suma acumulada por el número de iteraciones
average_confusion_matrix /= len(true_matrices_files)

# Imprimir la matriz promedio de confusión
print("Matriz Promedio de Confusión:")
print(average_confusion_matrix)
