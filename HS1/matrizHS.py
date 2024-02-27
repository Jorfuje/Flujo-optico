import cv2
import numpy as np

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

window = "Flujo óptico"
video_path = "HS1/video.mp4"
output_video_path = "videos/HSF.mp4"

capture = cv2.VideoCapture(video_path)

# Obtener la duración total del video original
fps = capture.get(cv2.CAP_PROP_FPS)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Obtener el tamaño del video de entrada
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear un objeto VideoWriter para el video resultante
fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

cv2.namedWindow(window)

ret, old_frame = capture.read()

scale_factor = 0.5
threshold = 1.5

while True:
    ret, frame = capture.read()

    if not ret:
        break

    u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

    # Dibujar los vectores
    for y in range(5, frame.shape[0] - 3, 10):
        for x in range(5, frame.shape[1] - 3, 10):
            magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)
            if magnitude > 6:
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 0, 255), 1, tipLength=0.5)
            elif magnitude < 6:
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

    cv2.imshow(window, frame)
    output_video.write(frame)

    old_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

capture.release()
output_video.release()
cv2.destroyAllWindows()

""" while True:
    ret, frame = capture.read()

    if not ret:
        break

    u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

    # Dibujar los vectores
    for y in range(5, frame.shape[0] - 3, 10):
        for x in range(5, frame.shape[1] - 3, 10):
            magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)
            if magnitude > 4.7:
                cell_x = x // 10
                cell_y = y // 10
                cv2.rectangle(frame, (cell_x * 10, cell_y * 10), ((cell_x + 1) * 10, (cell_y + 1) * 10), (0, 0, 255), -1)
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (255, 0, 0), 1, tipLength=0.5)
            elif magnitude < 4.7:
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

    cv2.imshow(window, frame)
    output_video.write(frame)

    old_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break """

""" while True:
    ret, frame = capture.read()

    if not ret:
        break

    u, v = horn_schunck_real_time(old_frame, frame, 0.01, 1)

    # Dibujar los vectores
    for y in range(5, frame.shape[0] - 3, 10):
        for x in range(5, frame.shape[1] - 3, 10):
            magnitude = np.sqrt(u[y, x] ** 2 + v[y, x] ** 2)
            if magnitude > 5:
                cell_x = x // 10
                cell_y = y // 10
                cv2.rectangle(frame, (cell_x * 10, cell_y * 10), ((cell_x + 1) * 10, (cell_y + 1) * 10), (0, 0, 255), -1)
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (255, 0, 0), 1, tipLength=0.5)
            elif magnitude < 5:
                cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

    cv2.imshow(window, frame)
    output_video.write(frame)

    old_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break """


""" capture.release()
output_video.release()
cv2.destroyAllWindows() """
