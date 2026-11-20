# camera_settings.py - Prueba: que ajustes de la camara acepta el driver (Fase 2, paso 1)
#
# Uso:  python camera_settings.py
# Objetivo: fijar el balance de blancos y la exposicion para que el tono de la imagen
# no cambie solo. En pantalla se ve el Cr y Cb medios de la imagen: con una pared blanca
# o gris, lo neutro es 128 (si Cr sube de 135, la imagen tira a rosa).
#
# Teclas: w     -> balance de blancos automatico ON/OFF
#         + / - -> temperatura de color (solo con balance manual)
#         x     -> exposicion automatica ON/OFF
#         e / d -> subir / bajar la exposicion (solo con exposicion manual)
#         p     -> abrir la ventana de ajustes del driver (Windows)
#         q     -> salir (imprime los valores finales)
#
# Cada cambio imprime en la terminal si el driver lo ha aceptado o no.

import os
import time
import cv2
import numpy as np

CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480
N_FPS = 30

PROPS = [
    ("AUTO_WB", cv2.CAP_PROP_AUTO_WB),
    ("WB_TEMPERATURE", cv2.CAP_PROP_WB_TEMPERATURE),
    ("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
    ("GAIN", cv2.CAP_PROP_GAIN),
    ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
]

# En DirectShow (Windows) la exposicion automatica usa 0.25 = manual y 0.75 = automatica
if os.name == "nt":
    EXPO_MANUAL, EXPO_AUTO = 0.25, 0.75
else:
    EXPO_MANUAL, EXPO_AUTO = 1, 3


def open_camera():
    if os.name == "nt":
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def show_props(cap, title):
    print("\n--- %s ---" % title)
    for name, prop in PROPS:
        print("  %-15s %s" % (name, cap.get(prop)))


def try_set(cap, name, prop, value):
    """Cambia un ajuste y comprueba si el driver lo ha aceptado."""
    before = cap.get(prop)
    cap.set(prop, value)
    after = cap.get(prop)
    accepted = abs(after - value) < 1e-3
    print("%-15s pedido %-8s | antes %-8s -> despues %-8s | %s"
          % (name, value, before, after, "ACEPTADO" if accepted else "NO CAMBIA"))
    return after


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolucion real: %dx%d" % (w, h))
    show_props(cap, "valores iniciales")

    wb_auto = True
    expo_auto = True
    times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break

        # FPS sobre los ultimos N_FPS frames
        times.append(time.time())
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        # Tono medio de la imagen: 128 = neutro
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr = ycc[:, :, 1].mean()
        cb = ycc[:, :, 2].mean()

        cv2.putText(frame, "FPS: %.1f   Cr: %.0f  Cb: %.0f  (neutro = 128)" % (fps, cr, cb),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "WB: %s   EXPO: %s" % ("auto" if wb_auto else "manual",
                                                  "auto" if expo_auto else "manual"),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("ajustes de camara", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            wb_auto = not wb_auto
            try_set(cap, "AUTO_WB", cv2.CAP_PROP_AUTO_WB, 1 if wb_auto else 0)
        elif key == ord("+") or key == ord("-"):
            step = 200 if key == ord("+") else -200
            try_set(cap, "WB_TEMPERATURE", cv2.CAP_PROP_WB_TEMPERATURE,
                    cap.get(cv2.CAP_PROP_WB_TEMPERATURE) + step)
        elif key == ord("x"):
            expo_auto = not expo_auto
            try_set(cap, "AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE,
                    EXPO_AUTO if expo_auto else EXPO_MANUAL)
        elif key == ord("e") or key == ord("d"):
            step = 1 if key == ord("e") else -1
            try_set(cap, "EXPOSURE", cv2.CAP_PROP_EXPOSURE,
                    cap.get(cv2.CAP_PROP_EXPOSURE) + step)
        elif key == ord("p"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)      # ventana de ajustes del driver

    show_props(cap, "valores finales (apuntalos)")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
