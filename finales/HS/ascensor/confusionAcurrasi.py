import numpy as np
import os

# Función para cargar una matriz desde un archivo de texto
def load_matrix_from_file(file_path):
    return np.loadtxt(file_path, dtype=int)

# Función para calcular la matriz de confusión y la precisión (accuracy)
def calculate_confusion_matrix_and_accuracy(true_matrix, predicted_matrix):
    # Inicializar la matriz de confusión
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    # Inicializar variables para calcular accuracy
    correct_predictions = 0
    total_predictions = 0

    # Comparar las matrices verdaderas y predichas
    for i in range(true_matrix.shape[0]):
        for j in range(true_matrix.shape[1]):
            if true_matrix[i, j] == 1:
                if predicted_matrix[i, j] == 1:
                    confusion_matrix[0, 0] += 1  # Verdadero positivo
                    correct_predictions += 1
                else:
                    confusion_matrix[0, 1] += 1  # Falso negativo
                total_predictions += 1
            else:
                if predicted_matrix[i, j] == 1:
                    confusion_matrix[1, 0] += 1  # Falso positivo
                else:
                    confusion_matrix[1, 1] += 1  # Verdadero negativo
                    correct_predictions += 1
                total_predictions += 1

    # Calcular la precisión (accuracy)
    accuracy = correct_predictions / total_predictions

    return confusion_matrix, accuracy

# Carpeta que contiene los archivos de matrices verdaderas
true_matrices_folder = "imagenes_salida/elevador/HS"
# Carpeta que contiene los archivos de matrices predichas
predicted_matrices_folder = "matrices_predichas/elevador"

# Obtener la lista de archivos en ambas carpetas
true_matrices_files = os.listdir(true_matrices_folder)
predicted_matrices_files = os.listdir(predicted_matrices_folder)

# Verificar que ambas carpetas tengan la misma cantidad de archivos
if len(true_matrices_files) != len(predicted_matrices_files):
    print("La cantidad de archivos en las carpetas no coincide. Verifica los archivos y vuelve a intentarlo.")
    exit()

# Inicializar la suma de matrices de confusión y accuracy
total_confusion_matrix = np.zeros((2, 2), dtype=int)
total_accuracy = 0.0

# Procesar cada par de archivos
for true_file_name, predicted_file_name in zip(true_matrices_files, predicted_matrices_files):
    # Construir las rutas completas de los archivos
    true_file_path = os.path.join(true_matrices_folder, true_file_name)
    predicted_file_path = os.path.join(predicted_matrices_folder, predicted_file_name)

    # Cargar las matrices verdadera y predicha desde los archivos
    true_matrix = load_matrix_from_file(true_file_path)
    predicted_matrix = load_matrix_from_file(predicted_file_path)

    # Calcular la matriz de confusión y la precisión (accuracy)
    confusion_matrix, accuracy = calculate_confusion_matrix_and_accuracy(true_matrix, predicted_matrix)

    # Sumar la matriz de confusión y la precisión (accuracy) total
    total_confusion_matrix += confusion_matrix
    total_accuracy += accuracy

    # Imprimir la matriz de confusión y la precisión (accuracy) para esta iteración
    print(f"Matriz de Confusión para {true_file_name} vs {predicted_file_name}:")
    print(confusion_matrix)
    print(f"Accuracy para {true_file_name} vs {predicted_file_name}: {accuracy:.4f}")
    print()

# Calcular el promedio de la matriz de confusión y la precisión (accuracy)
average_confusion_matrix = total_confusion_matrix / len(true_matrices_files)
average_accuracy = total_accuracy / len(true_matrices_files)

# Imprimir la matriz de confusión y la precisión (accuracy) promedio
print("Matriz de Confusión Promedio:")
print(average_confusion_matrix)
print(f"Accuracy Promedio: {average_accuracy:.4f}")
