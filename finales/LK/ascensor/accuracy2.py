import numpy as np
import os
import csv

def load_confusion_matrix(file_path):
    return np.loadtxt(file_path, dtype=int)

def calculate_accuracy(confusion_matrix):
    a = tp = confusion_matrix[0, 0]         # Verdadero positivo
    b = fn = confusion_matrix[0, 1]         # Falso negativo
    c = fp = confusion_matrix[1, 0]         # Falso positivo
    d = tn = confusion_matrix[1, 1]         # Verdadero negativo
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # accuracy1 = (a + d) / (a + b + c + d)
    # print(accuracy, accuracy1)
    
    return accuracy

def calculate_precision(confusion_matrix):
    a = confusion_matrix[0, 0]         # Verdadero positivo
    c = confusion_matrix[1, 0]         # Falso positivo
    
    denominator = a + c
    if denominator == 0:
        return 0  # O cualquier otro valor que desees retornar en este caso especial
    
    precision = a / denominator
    return precision

def calculate_recall(confusion_matrix):
    a = confusion_matrix[0, 0]         # Verdadero positivo
    b = confusion_matrix[0, 1]         # Falso negativo
    
    denominator = a + b
    if denominator == 0:
        return 0  # O cualquier otro valor que desees retornar en este caso especial
    
    recall = a / denominator
    return recall

def calculate_measure(confusion_matrix):
    a = confusion_matrix[0, 0]         # Verdadero positivo
    b = confusion_matrix[0, 1]         # Falso negativo
    c = confusion_matrix[1, 0]         # Falso positivo
    
    denominator = 2 * a + b + c
    if denominator == 0:
        return 0  # O cualquier otro valor que desees retornar en este caso especial
    
    measure = 2 * a / denominator
    return measure


confusion_matrices_folder = "matrices_confusion/elevador/LK"
accuracy_matrix_file = "metricas/elevador/LK/accuracy_matrixLK.txt"

confusion_matrix_files = os.listdir(confusion_matrices_folder)

accuracy_matrix = {}    # Nuevo diccionario para almacenar las accuracy
precision_matrix = {}   # Nuevo diccionario para almacenar las precisiones
recall_matrix = {}   # Nuevo diccionario para almacenar las recall
measure_matrix = {}   # Nuevo diccionario para almacenar las measure

for file_name in confusion_matrix_files:
    file_path = os.path.join(confusion_matrices_folder, file_name)
    confusion_matrix = load_confusion_matrix(file_path)
    accuracy = calculate_accuracy(confusion_matrix)     # Calcular accuracy
    precision = calculate_precision(confusion_matrix)   # Calcular precisión
    recall = calculate_recall(confusion_matrix)   # Calcular recall
    measure = calculate_measure(confusion_matrix)   # Calcular measure
    
    # Extraer el número de iteración del nombre del archivo
    iteracion = file_name.split("_")[-1]
    iteracion = iteracion.split(".")[0]  # Eliminar la extensión del archivo
    
    # Guardar la precisión en el diccionario con el nombre del archivo como clave
    accuracy_matrix[file_name] = accuracy       # Guardar accuracy en el nuevo diccionario
    precision_matrix[file_name] = precision     # Guardar precisión en el nuevo diccionario
    recall_matrix[file_name] = recall           # Guardar recall en el nuevo diccionario
    measure_matrix[file_name] = measure           # Guardar measure en el nuevo diccionario

# Ordenar los diccionarios por número de iteración
accuracy_matrix_sorted = dict(sorted(accuracy_matrix.items(), key=lambda x: int(x[0].split("_")[-1].split(".")[0])))
precision_matrix_sorted = dict(sorted(precision_matrix.items(), key=lambda x: int(x[0].split("_")[-1].split(".")[0])))
recall_matrix_sorted = dict(sorted(recall_matrix.items(), key=lambda x: int(x[0].split("_")[-1].split(".")[0])))
measure_matrix_sorted = dict(sorted(measure_matrix.items(), key=lambda x: int(x[0].split("_")[-1].split(".")[0])))

# Extraer solo los valores de precisión
accuracy_values = list(accuracy_matrix_sorted.values())
precision_values = list(precision_matrix_sorted.values())  # Extraer solo los valores de precisión
recall_values = list(recall_matrix_sorted.values())  # Extraer solo los valores de recall
measure_values = list(measure_matrix_sorted.values())  # Extraer solo los valores de measure

# Guardar los valores de precisión en un nuevo documento llamado "metricas"
with open('metricas/elevador/LK/metricasLK.txt', 'w') as file:
    file.write("accuracyLK = " + str(accuracy_values) + "\n")
    file.write("precisionLK = " + str(precision_values) + "\n")
    file.write("recallLK = " + str(recall_values) + "\n")
    file.write("measureLK = " + str(measure_values) + "\n")

# Guardar la matriz de precisión en un archivo de texto
with open(accuracy_matrix_file, "w") as f:
    for file_name, accuracy in accuracy_matrix_sorted.items():
        precision = precision_matrix_sorted[file_name]  # Obtener la precisión correspondiente
        recall = recall_matrix_sorted[file_name]  # Obtener la recall correspondiente
        measure = measure_matrix_sorted[file_name]  # Obtener la measure correspondiente
        
        f.write(f"{file_name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Measure = {measure:.4f}\n")

# Convertir los valores de las métricas en arrays de NumPy
accuracy_values = np.array(accuracy_values)
precision_values = np.array(precision_values)
recall_values = np.array(recall_values)
measure_values = np.array(measure_values)

# Mínimo
accuracy_min = np.min(accuracy_values)
precision_min = np.min(precision_values)
recall_min = np.min(recall_values)
measure_min = np.min(measure_values)

# Máximo
accuracy_max = np.max(accuracy_values)
precision_max = np.max(precision_values)
recall_max = np.max(recall_values)
measure_max = np.max(measure_values)

# Media
accuracy_mean = np.mean(accuracy_values)
precision_mean = np.mean(precision_values)
recall_mean = np.mean(recall_values)
measure_mean = np.mean(measure_values)

# Mediana
accuracy_median = np.median(accuracy_values)
precision_median = np.median(precision_values)
recall_median = np.median(recall_values)
measure_median = np.median(measure_values)

# Desviación estándar
accuracy_std = np.std(accuracy_values)
precision_std = np.std(precision_values)
recall_std = np.std(recall_values)
measure_std = np.std(measure_values)

# Definir los datos
metricas = {
    "Métrica": ["Accuracy", "Precision", "Recall", "F1-Measure"],
    "Mínimo": [accuracy_min, precision_min, recall_min, measure_min],
    "Máximo": [accuracy_max, precision_max, recall_max, measure_max],
    "Media": [accuracy_mean, precision_mean, recall_mean, measure_mean],
    "Mediana": [accuracy_median, precision_median, recall_median, measure_median],
    "Desviación Estándar": [accuracy_std, precision_std, recall_std, measure_std]
}

# Nombre del archivo de salida
archivo_salida = "metricas/elevador/LK/metricasLK.csv"

# Escribir los datos en el archivo CSV
with open(archivo_salida, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Métrica", "Mínimo", "Máximo", "Media", "Mediana", "Desviación Estándar"])
    
    # Escribir el encabezado
    writer.writeheader()
    
    # Escribir los datos
    for i in range(len(metricas["Métrica"])):
        writer.writerow({
            "Métrica": metricas["Métrica"][i],
            "Mínimo": metricas["Mínimo"][i],
            "Máximo": metricas["Máximo"][i],
            "Media": metricas["Media"][i],
            "Mediana": metricas["Mediana"][i],
            "Desviación Estándar": metricas["Desviación Estándar"][i]
        })

print(f"Los datos de métricas se han guardado en el archivo: {archivo_salida}")

print("Matriz de Precisión (Accuracy), Precisión, Recall y Measure Guardadas en:", accuracy_matrix_file)
