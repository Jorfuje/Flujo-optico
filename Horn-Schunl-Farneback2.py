import cv2
import numpy as np

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

if __name__ == "__main__":
    video_path = "video.mp4"  # Cambia 'ruta_del_video.mp4' con la ubicación de tu video
    capture = cv2.VideoCapture(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crear un objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if cv2.VideoWriter_fourcc(*'mp4v') != -1 else cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = "videos/HSF.mp4"
    output_video = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        u, v = horn_schunck_real_time(old_frame, frame, 0.005, 1)

        # Ajusta los parámetros según sea necesario para obtener una visualización más clara
        scale_factor = .5
        threshold = 0.1

        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                if np.abs(u[y, x]) > threshold or np.abs(v[y, x]) > threshold:
                    cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

        cv2.imshow("Flujo óptico", frame)
        output_video.write(frame)

        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
