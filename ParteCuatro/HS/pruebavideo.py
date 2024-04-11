import cv2

# Leer el video
video_path = "ParteCuatro/HS/output.mp4"
capture = cv2.VideoCapture(video_path)

# Obtener la cantidad de frames
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Procesar el video
frame_count = 0
while True:
    # Leer el frame actual
    ret, frame = capture.read()

    # Si no se pudo leer el frame, salir del bucle
    if not ret:
        break

    # Procesar el frame (por ejemplo, convertirlo a escala de grises)
    # ...

    # Incrementar el contador de frames
    frame_count += 1

# Mostrar la cantidad de frames
print(f"Total de frames: {total_frames}")
print(f"Cantidad de frames: {frame_count}")

# Liberar recursos
capture.release()
cv2.destroyAllWindows()
