import cv2

# Ruta al archivo de video
video_path = "LK.mp4"

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Contador de fotogramas
frame_count = 0

# Leer hasta que se alcance el final del video
while(cap.isOpened()):
    # Leer un fotograma del video
    ret, frame = cap.read()

    # Verificar si se ha alcanzado el final del video
    if not ret:
        break

    # Guardar el fotograma en un archivo de imagen
    frame_filename = f"frame_{frame_count:04d}.jpg"  # Nombre del archivo de imagen
    cv2.imwrite(frame_filename, frame)  # Guardar el fotograma como imagen

    # Incrementar el contador de fotogramas
    frame_count += 1

    # Mostrar el progreso
    print(f"Frame {frame_count} guardado")

# Liberar los recursos
cap.release()

print("Todos los fotogramas han sido guardados.")
