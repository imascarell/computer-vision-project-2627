# capture.py - Captura de imagenes del tablero para la calibracion (Fase 1, paso 4)
#
# Uso:  python capture.py
# Teclas: s -> guardar imagen (solo si se detectan las esquinas)
#         q -> salir
#
# Las imagenes se guardan en calibration/images/ sin dibujos encima.

import glob
import os
import cv2

CAM_INDEX = 0
WIDTH = 640                 # misma resolucion que en la demo
HEIGHT = 480
PATTERN = (6, 8)            # esquinas interiores del tablero impreso (7x9 cuadros)
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolucion real: %dx%d" % (w, h))
    if (w, h) != (WIDTH, HEIGHT):
        print("AVISO: no es la resolucion esperada, no uses estas imagenes")

    n = len(glob.glob(os.path.join(IMAGES_DIR, "img_*.png")))   # ya guardadas
    print("Imagenes ya guardadas: %d" % n)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PATTERN, FLAGS)

        view = frame.copy()                      # sobre esta copia se dibuja
        if found:
            cv2.drawChessboardCorners(view, PATTERN, corners, found)
            msg, color = "Tablero detectado: pulsa 's'", (0, 255, 0)
        else:
            msg, color = "Tablero NO detectado", (0, 0, 255)
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(view, "Guardadas: %d" % n, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("captura", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and found:
            n = n + 1
            path = os.path.join(IMAGES_DIR, "img_%02d.png" % n)
            while os.path.exists(path):          # no sobrescribir
                n = n + 1
                path = os.path.join(IMAGES_DIR, "img_%02d.png" % n)
            cv2.imwrite(path, frame)             # imagen limpia, sin esquinas
            print("Guardada:", path)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
