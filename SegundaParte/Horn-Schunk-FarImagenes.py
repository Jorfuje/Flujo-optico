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
    image_path1 = "SegundaParte/image1.jpg"  # Ruta de la primera imagen
    image_path2 = "SegundaParte/image2.jpg"  # Ruta de la segunda imagen

    # Leer las imágenes
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Obtener las dimensiones de las imágenes
    height, width, _ = image1.shape

    # Aplicar el algoritmo de Horn-Schunck
    u, v = horn_schunck_real_time(image1, image2, 0.005, 1)

    # Ajustar los parámetros según sea necesario para obtener una visualización más clara
    scale_factor = .5
    threshold = 1.5

    for y in range(0, height, 10):
        for x in range(0, width, 10):
            if np.abs(u[y, x]) > threshold or np.abs(v[y, x]) > threshold:
                cv2.arrowedLine(image2, (x, y), (int(x + scale_factor * u[y, x]), int(y + scale_factor * v[y, x])), (0, 255, 0), 1, tipLength=0.5)

    # Guardar la imagen resultante como archivo de imagen
    output_image_path = "output_imageHS.jpg"
    cv2.imwrite(output_image_path, image2)
    print("Imagen guardada como:", output_image_path)

    # Mostrar la imagen resultante en una ventana
    cv2.imshow("Imagen resultante", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

