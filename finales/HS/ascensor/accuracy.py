import numpy as np
import os

def load_confusion_matrix(file_path):
    return np.loadtxt(file_path, dtype=int)

def calculate_accuracy(confusion_matrix):
    tp = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    tn = confusion_matrix[1, 1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

confusion_matrices_folder = "matrices_confusion/elevador/HS"
accuracy_matrix_file = "metricas/elevador/HS/accuracy_matrix.txt"

confusion_matrix_files = os.listdir(confusion_matrices_folder)

accuracy_matrix = {}

for file_name in confusion_matrix_files:
    file_path = os.path.join(confusion_matrices_folder, file_name)
    confusion_matrix = load_confusion_matrix(file_path)
    accuracy = calculate_accuracy(confusion_matrix)
    
    # Extraer el número de iteración del nombre del archivo
    iteracion = file_name.split("_")[-1]
    iteracion = iteracion.split(".")[0]  # Eliminar la extensión del archivo
    
    # Guardar la precisión en el diccionario con el nombre del archivo como clave
    accuracy_matrix[file_name] = accuracy

# Ordenar el diccionario por número de iteración
accuracy_matrix_sorted = dict(sorted(accuracy_matrix.items(), key=lambda x: int(x[0].split("_")[-1].split(".")[0])))

# Extraer solo los valores de precisión
accuracy_values = list(accuracy_matrix_sorted.values())

# Guardar los valores en un nuevo documento llamado "metricas"
with open('metricas/elevador/HS/metricas.txt', 'w') as file:
    file.write("accuracy = " + str(accuracy_values))


# Guardar la matriz de precisión en un archivo de texto
with open(accuracy_matrix_file, "w") as f:
    for file_name, accuracy in accuracy_matrix_sorted.items():
        f.write(f"{file_name}: {accuracy:.4f}\n")

print("Matriz de Precisión (Accuracy) Guardada en:", accuracy_matrix_file)
