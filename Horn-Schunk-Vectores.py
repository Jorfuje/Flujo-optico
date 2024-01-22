import cv2
import numpy as np

def horn_schunck_real_time(I1, I2, alpha, iterations):
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    u = np.zeros_like(I1_gray)
    v = np.zeros_like(I1_gray)

    for _ in range(iterations):
        Ix = cv2.Sobel(I1_gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(I1_gray, cv2.CV_64F, 0, 1, ksize=5)
        It = I2_gray - I1_gray

        u_avg = cv2.blur(u, (3, 3))
        v_avg = cv2.blur(v, (3, 3))

        du = -alpha * (Ix * (Ix * u_avg + Iy * v_avg + It))
        dv = -alpha * (Iy * (Ix * u_avg + Iy * v_avg + It))

        u += du
        v += dv

    return u, v

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("Flujo óptico")

    ret, old_frame = capture.read()

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        u, v = horn_schunck_real_time(old_frame, frame, 0.005, 2)

        # Ajusta la escala y grosor de los vectores
        scale_factor = 500
        thickness = 1

        # Dibuja los vectores con líneas más largas y cabezas más grandes
        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                start_point = (x, y)
                end_point = (int(x + u[y, x] * scale_factor), int(y + v[y, x] * scale_factor))
                color = (0, 255, 0)
                cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=.5)

        cv2.imshow("Flujo óptico", frame)
        old_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()








