import cv2
import numpy as np

def horn_schunck_real_time(I1, I2, alpha, iterations):
    # Convierte las imágenes a escala de grises
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Calcula el flujo óptico con calcOpticalFlowFarneback
    flow = cv2.calcOpticalFlowFarneback(I1_gray, I2_gray, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    # Extrae los componentes u y v del flujo
    u = flow[:,:,0]
    v = flow[:,:,1]

    return u, v

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        # Calcula el flujo óptico
        u, v = horn_schunck_real_time(old_frame, frame, 0.02, 1)

        # Dibuja los vectores de flujo óptico
        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                cv2.arrowedLine(frame, (x, y), (int(x + u[y, x]), int(y + v[y, x])), (0, 255, 0), 2, tipLength=0.5)

        cv2.imshow("Flujo óptico", frame)
        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
