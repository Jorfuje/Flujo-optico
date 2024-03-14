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

def sumar_cinco_a_promedios(promedio_file):
    with open(promedio_file, 'r') as file:
        lines = file.readlines()
    
    new_promedios = []
    for line in lines:
        promedio = float(line.split(':')[-1].strip())
        new_promedio = promedio + 11
        print(new_promedio)
        new_promedios.append(new_promedio)
    
    return new_promedios

if __name__ == "__main__":
    video_path = "HS1/caida.mp4"
    capture = cv2.VideoCapture(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = "videos/HSF.mp4"
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    scale_factor = 0.5
    threshold = 4
    promedio_file = "promedioCeldas.txt"
    new_promedios = sumar_cinco_a_promedios(promedio_file)

    grid_size = 20  # Tamaño de la cuadrícula 20x20
    grid_width = width // grid_size
    grid_height = height // grid_size

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

        for y in range(0, height, grid_height):
            for x in range(0, width, grid_width):
                if y + grid_height <= height and x + grid_width <= width:
                    magnitude = np.sqrt(np.sum(u[y:y+grid_height, x:x+grid_width])**2 + np.sum(v[y:y+grid_height, x:x+grid_width])**2)
                    promedio_celda = new_promedios[(y // grid_height) * grid_size + (x // grid_width)]

                    if magnitude >= promedio_celda:
                        color = (0, 0, 255)  # Rojo
                    else:
                        color = (0, 255, 0)  # Verde

                    cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), color, 1, tipLength=0.5)
                    cv2.rectangle(frame, (x, y), (x + grid_width, y + grid_height), (0, 255, 255), 1)

        cv2.imshow("Flujo óptico", frame)
        output_video.write(frame)

        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
