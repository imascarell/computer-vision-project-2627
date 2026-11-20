# camera.py - Lectura de la camara con FPS en pantalla (Fase 1, paso 2)
#
# Uso:  python camera.py
# Teclas: q -> salir

import time
import cv2

CAM_INDEX = 0      # 0 = camara del portatil
WIDTH = 640        # resolucion fija (la misma en calibracion y demo)
HEIGHT = 480
N_FPS = 30         # frames que se promedian para calcular los FPS


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    # La camara puede no aceptar la resolucion pedida: se comprueba la real
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolucion pedida: %dx%d | real: %dx%d" % (WIDTH, HEIGHT, w, h))

    times = []                 # instantes de los ultimos frames
    start = time.time()
    total_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break

        # FPS = frames / tiempo, sobre los ultimos N_FPS frames
        now = time.time()
        times.append(now)
        if len(times) > N_FPS:
            times.pop(0)
        total_frames = total_frames + 1
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        cv2.putText(frame, "FPS: %.1f" % fps, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - start
    cap.release()
    cv2.destroyAllWindows()
    if elapsed > 0:
        print("FPS medios de toda la ejecucion: %.1f" % (total_frames / elapsed))


if __name__ == "__main__":
    main()
