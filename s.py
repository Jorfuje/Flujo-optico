import cv2
import numpy as np

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    flow = cv2.calcOpticalFlowHS(I1_gray, I2_gray, None, alpha, (iterations, iterations), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    return u, v

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        u, v = horn_schunck_real_time(old_frame, frame, 0.02, 1)

        # Ajusta los parámetros según sea necesario para obtener una visualización más clara
        scale_factor = 5
        threshold = 0.1

        for y in range(0, frame.shape[0], 20):
            for x in range(0, frame.shape[1], 20):
                if np.abs(u[y, x]) > threshold or np.abs(v[y, x]) > threshold:
                    cv2.arrowedLine(frame, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

        cv2.imshow("Flujo óptico", frame)
        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
