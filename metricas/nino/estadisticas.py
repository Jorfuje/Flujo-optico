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
# round(numero, 4)
h2A = round(normaltest(difAccuracy).pvalue, 4)
shaA = round(shapiro(difAccuracy).pvalue, 4)
skeA = round(skewtest(difAccuracy).pvalue, 4)
kurA = round(kurtosistest(difAccuracy).pvalue, 4)
ttestA = round(ttest_rel(accuracyHS, accuracyLK).pvalue, 4)
wilA = round(wilcoxon(difAccuracy).pvalue, 4)

# Precision
h2P = round(normaltest(difPrecision).pvalue, 4)
shaP = round(shapiro(difPrecision).pvalue, 4)
skeP = round(skewtest(difPrecision).pvalue, 4)
kurP = round(kurtosistest(difPrecision).pvalue, 4)
ttestP = round(ttest_rel(precisionHS, precisionLK).pvalue, 4)
wilP = round(wilcoxon(difPrecision).pvalue, 4)

# Recall
h2R = round(normaltest(difRecall).pvalue, 4)
shaR = round(shapiro(difRecall).pvalue, 4)
skeR = round(skewtest(difRecall).pvalue, 4)
kurR = round(kurtosistest(difRecall).pvalue, 4)
ttestR = round(ttest_rel(recallHS, recallLK).pvalue, 4)
wilR = round(wilcoxon(difRecall).pvalue, 4)

# F1-measure
h2F = round(normaltest(difMeasure).pvalue, 4)
shaF = round(shapiro(difMeasure).pvalue, 4)
skeF = round(skewtest(difMeasure).pvalue, 4)
kurF = round(kurtosistest(difMeasure).pvalue, 4)
ttestF = round(ttest_rel(measureHS, measureLK).pvalue, 4)
wilF = round(wilcoxon(difMeasure).pvalue, 4)



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
archivo_salida = "metricas/nino/estadisticasNino.csv"

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