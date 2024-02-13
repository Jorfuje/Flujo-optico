import cv2
import os

# Ruta al archivo de video
video_path = "Herramientas/video.mp4"
#video_path = "videos/HSF.mp4"

# Carpeta donde se guardarán las imágenes
output_folder = "imagenes1"
#output_folder = "imagenes2"
#output_folder = "imagenes3"

# Crear la carpeta si no existe
os.makedirs(output_folder, exist_ok=True)

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
    frame_filename = os.path.join(output_folder, f"image_{frame_count:04d}.jpg")  # Nombre del archivo de imagen
    cv2.imwrite(frame_filename, frame)  # Guardar el fotograma como imagen

    # Incrementar el contador de fotogramas
    frame_count += 1

    # Mostrar el progreso
    print(f"Frame {frame_count} guardado")

# Liberar los recursos
cap.release()

print("Todos los fotogramas han sido guardados en la carpeta 'imagenes'.")

