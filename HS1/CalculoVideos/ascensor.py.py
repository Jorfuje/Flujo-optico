import cv2
import numpy as np
import os

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

def sumar_cinco_a_promedios(promedio_file, grid_size):
    with open(promedio_file, 'r') as file:
        lines = file.readlines()
    
    new_promedios = []
    for line in lines:
        promedio = float(line.split(':')[-1].strip())
        new_promedio = promedio + 5.97
        new_promedios.append(new_promedio)
    
    # Ajustar el tamaño de new_promedios si es necesario
    expected_length = grid_size ** 2
    if len(new_promedios) < expected_length:
        new_promedios.extend([0] * (expected_length - len(new_promedios)))

    return new_promedios

if __name__ == "__main__":
    video_path = "HS1/cortado.mp4"
    capture = cv2.VideoCapture(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)  # Obtener FPS del video original

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = "videos/HSF.mp4"
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # Establecer la misma FPS para el video de salida

    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    scale_factor = 0.5
    threshold = 4

    promedio_file = "promedioCeldas.txt"
    grid_size = 20
    new_promedios = sumar_cinco_a_promedios(promedio_file, grid_size)

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)
                promedio_celda = new_promedios[(y // 10) * (frame.shape[1] // 10) + (x // 10)]  # Promedio de la celda

                if magnitude >= promedio_celda:
                    color = (0, 0, 255)  # Rojo
                    cv2.rectangle(frame, (x, y), (x + 10, y + 10), color, -1)  # Rellenar celda en rojo
                else:
                    color = (0, 255, 0)  # Verde

                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), color, 1, tipLength=0.5)
                cv2.rectangle(frame, (x, y), (x + 10, y + 10), (0, 255, 255), 1)  # Dibujar cuadrícula

        cv2.imshow("Flujo óptico", frame)
        output_video.write(frame)

        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
