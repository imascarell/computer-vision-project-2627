# camera.py - Camara con correccion de distorsion en vivo y FPS (Fase 1, pasos 2, 6 y 7)
#
# Uso:  python camera.py
# Teclas: u -> alterna imagen corregida / original
#         s -> guarda original, corregida y diferencia amplificada en security/captures/
#         q -> salir
#
# Al salir se imprimen los FPS medios con la correccion activada y desactivada.

import glob
import os
import time
import cv2
import numpy as np

CAM_INDEX = 0
WIDTH = 640                # misma resolucion que en la calibracion
HEIGHT = 480
N_FPS = 30                 # frames que se promedian para los FPS en pantalla
ALPHA = 0                  # 0 = la imagen corregida no tiene bordes negros
DIFF_GAIN = 40             # amplificacion de la imagen de diferencia (la correccion es de ~2 px)

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "..", "calibration", "calib.npz")
CAPTURES_DIR = os.path.join(HERE, "captures")


def load_maps():
    """Lee la calibracion y calcula los mapas de correccion (una sola vez)."""
    data = np.load(CALIB_FILE)
    K = data["K"]
    dist = data["dist"]
    size = (int(data["img_size"][0]), int(data["img_size"][1]))      # (ancho, alto)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, ALPHA)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)
    return size, map1, map2


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def save_capture(frame, corrected):
    """Guarda original, corregida y diferencia amplificada (para el informe)."""
    diff = cv2.absdiff(frame, corrected)
    diff = cv2.convertScaleAbs(diff, alpha=DIFF_GAIN)
    n = len(glob.glob(os.path.join(CAPTURES_DIR, "undistort_*_original.png"))) + 1
    base = os.path.join(CAPTURES_DIR, "undistort_%02d" % n)
    cv2.imwrite(base + "_original.png", frame)
    cv2.imwrite(base + "_corregida.png", corrected)
    cv2.imwrite(base + "_diferencia.png", diff)
    print("Guardadas las 3 imagenes: %s_*.png" % base)


def main():
    size, map1, map2 = load_maps()

    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolucion pedida: %dx%d | real: %dx%d | calibracion: %dx%d"
          % (WIDTH, HEIGHT, w, h, size[0], size[1]))
    if (w, h) != size:
        # K depende de la resolucion: con otra no vale
        print("ERROR: la resolucion de la camara no coincide con la de la calibracion")
        cap.release()
        return

    undistort_on = True
    times = []                              # instantes de los ultimos frames
    frames_mode = {True: 0, False: 0}       # frames y tiempo acumulado por modo
    time_mode = {True: 0.0, False: 0.0}
    prev = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break

        # Correccion de distorsion: un remap por frame
        if undistort_on:
            view = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        else:
            view = frame.copy()

        # FPS = frames / tiempo, sobre los ultimos N_FPS frames
        now = time.time()
        times.append(now)
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])
        if prev is not None:
            frames_mode[undistort_on] += 1
            time_mode[undistort_on] += now - prev
        prev = now

        label = "CORREGIDA" if undistort_on else "ORIGINAL"
        cv2.putText(view, "FPS: %.1f  %s" % (fps, label), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("u"):
            undistort_on = not undistort_on
        elif key == ord("s"):
            save_capture(frame, cv2.remap(frame, map1, map2, cv2.INTER_LINEAR))

    cap.release()
    cv2.destroyAllWindows()
    for mode, name in ((True, "corregida"), (False, "original")):
        if time_mode[mode] > 0:
            print("FPS medios (%s): %.1f sobre %d frames"
                  % (name, frames_mode[mode] / time_mode[mode], frames_mode[mode]))


if __name__ == "__main__":
    main()
