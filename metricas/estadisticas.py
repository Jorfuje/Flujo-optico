from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import skewtest
from scipy.stats import kurtosistest
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
import re
import csv

# Abre el archivo en modo de lectura
with open('metricas/nino/metricasNino.txt', 'r') as archivo:
    # Lee todo el contenido del archivo
    contenido = archivo.read()

# Define una función para extraer los arreglos de la cadena de texto
def obtener_arreglo(nombre_arreglo):
    # Busca el patrón del arreglo en la cadena de texto
    patron = re.compile(rf"{nombre_arreglo}\s*=\s*\[(.*?)\]", re.DOTALL)
    match = patron.search(contenido)
    if match:
        # Extrae la cadena de valores dentro de los corchetes
        valores = match.group(1)
        # Divide la cadena en una lista de valores numéricos
        arreglo = [float(valor) for valor in valores.split(',') if valor.strip()]
        return arreglo
    else:
        return None

# Llama a la función para obtener los arreglos específicos
accuracyHS = obtener_arreglo("accuracyHS")
accuracyLK = obtener_arreglo("accuracyLK")
precisionHS = obtener_arreglo("precisionHS")
precisionLK = obtener_arreglo("precisionLK")
recallHS = obtener_arreglo("recallHS")
recallLK = obtener_arreglo("recallLK")
measureHS = obtener_arreglo("measureHS")
measureLK = obtener_arreglo("measureLK")

# Accuracy
if len(accuracyHS) == len(accuracyLK):
    # Calcula la diferencia entre los elementos correspondientes de los arreglos
    difAccuracy = [hs - lk for hs, lk in zip(accuracyHS, accuracyLK)]
else:
    print("Los arreglos tienen diferentes longitudes y no se pueden restar.")
    
# Precision
if len(precisionHS) == len(precisionLK):
    # Calcula la diferencia entre los elementos correspondientes de los arreglos
    difPrecision = [hs - lk for hs, lk in zip(precisionHS, precisionLK)]
else:
    print("Los arreglos tienen diferentes longitudes y no se pueden restar.")

# Recall
if len(recallHS) == len(recallLK):
    # Calcula la diferencia entre los elementos correspondientes de los arreglos
    difRecall = [hs - lk for hs, lk in zip(recallHS, recallLK)]
else:
    print("Los arreglos tienen diferentes longitudes y no se pueden restar.")

# Measure
if len(measureHS) == len(measureLK):
    # Calcula la diferencia entre los elementos correspondientes de los arreglos
    difMeasure = [hs - lk for hs, lk in zip(measureHS, measureLK)]
else:
    print("Los arreglos tienen diferentes longitudes y no se pueden restar.")

# Accuracy
h2A = normaltest(difAccuracy).pvalue
shaA = shapiro(difAccuracy).pvalue
skeA = skewtest(difAccuracy).pvalue
kurA = kurtosistest(difAccuracy).pvalue
ttestA = ttest_rel(accuracyHS, accuracyLK).pvalue
wilA = wilcoxon(difAccuracy).pvalue

# Precision
h2P = normaltest(difPrecision).pvalue
shaP = shapiro(difPrecision).pvalue
skeP = skewtest(difPrecision).pvalue
kurP = kurtosistest(difPrecision).pvalue
ttestP = ttest_rel(precisionHS, precisionLK).pvalue
wilP = wilcoxon(difPrecision).pvalue

# Recall
h2R = normaltest(difRecall).pvalue
shaR = shapiro(difRecall).pvalue
skeR = skewtest(difRecall).pvalue
kurR = kurtosistest(difRecall).pvalue
ttestR = ttest_rel(recallHS, recallLK).pvalue
wilR = wilcoxon(difRecall).pvalue

# F1-measure
h2F = normaltest(difMeasure).pvalue
shaF = shapiro(difMeasure).pvalue
skeF = skewtest(difMeasure).pvalue
kurF = kurtosistest(difMeasure).pvalue
ttestF = ttest_rel(measureHS, measureLK).pvalue
wilF = wilcoxon(difMeasure).pvalue



# Definir los datos
metricas = {
    "Métrica": ["Accuracy", "Precision", "recall", "F1"],
    "x2": [h2A, h2P, h2R, h2F],
    "sw": [shaA, shaP, shaR, shaF],
    "skewness": [skeA, skeP, skeR, skeF],
    "kurtosis": [kurA, kurP, kurR, kurF],
    "ttest": [ttestA, ttestP, ttestR, ttestF],
    "wilcoxon": [wilA, wilP, wilR, wilF],
}

# Nombre del archivo de salida
archivo_salida = "metricas/estadisticas.csv"

# Escribir los datos en el archivo CSV
with open(archivo_salida, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Métrica", "x2", "sw", "skewness", "kurtosis", "ttest", "wilcoxon"])
    
    # Escribir el encabezado
    writer.writeheader()
    
    

# Escribir los datos
    for i in range(len(metricas["Métrica"])):
        writer.writerow({
            "Métrica": metricas["Métrica"][i],
            "x2": metricas["x2"][i],
            "sw": metricas["sw"][i],
            "skewness": metricas["skewness"][i],
            "kurtosis": metricas["kurtosis"][i],
            "ttest": metricas["ttest"][i],
            "wilcoxon": metricas["wilcoxon"][i],
        })