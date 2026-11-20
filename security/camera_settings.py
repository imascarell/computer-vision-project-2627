# camera_settings.py - Prueba: que ajustes de la camara acepta el driver (Fase 2, paso 1)
#
# Uso:  python camera_settings.py
# Objetivo: comprobar si el balance de blancos y la exposicion quedan FIJOS cuando entra
# la mano delante de la camara. Se mide un parche de pared (cuadrado verde, arriba a la
# izquierda): si el ajuste esta fijo, su brillo (Y) y su color (Cr, Cb) no cambian aunque
# muevas la mano por el resto de la imagen. Con una pared blanca o gris, Cr y Cb neutros = 128.
#
# IMPORTANTE: las teclas solo funcionan con la ventana de la camara seleccionada (haz clic en ella).
#
# Teclas: w / a -> balance de blancos MANUAL / AUTOMATICO
#         + / - -> temperatura de color (solo con balance manual)
#         x / z -> exposicion MANUAL / AUTOMATICA
#         e / d -> subir / bajar la exposicion (solo con exposicion manual)
#         l     -> medir la estabilidad del parche durante 10 s (termina sola)
#         p     -> abrir la ventana de ajustes del driver (Windows)
#         q     -> salir (imprime los valores finales)
#
# Los ajustes se quedan guardados en el driver entre ejecuciones: lo que se ve en pantalla
# en la linea "Driver" es el estado REAL de la camara, no el que tu creas.

import os
import time
import cv2
import numpy as np

CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480
N_FPS = 30
PATCH = (10, 10, 90, 90)            # parche de pared: x, y, ancho, alto (arriba a la izquierda)
MEASURE_SECONDS = 10               # duracion de la medicion de estabilidad
MAX_RANGE_Y = 6                     # variacion maxima (max - min) para dar el parche por estable
MAX_RANGE_CRCB = 3

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


def report_stability(samples):
    """Resume la medicion del parche: variacion de Y, Cr y Cb."""
    data = np.array(samples)                       # filas: (Y, Cr, Cb)
    print("\nMedicion de estabilidad: %d frames" % len(data))
    names = ["Y (brillo)", "Cr", "Cb"]
    ranges = []
    for i in range(3):
        col = data[:, i]
        ranges.append(col.max() - col.min())
        print("  %-10s media %6.1f | min %6.1f | max %6.1f | variacion %5.1f"
              % (names[i], col.mean(), col.min(), col.max(), ranges[-1]))
    stable = ranges[0] <= MAX_RANGE_Y and ranges[1] <= MAX_RANGE_CRCB and ranges[2] <= MAX_RANGE_CRCB
    print("  -> %s (limites: Y <= %d, Cr y Cb <= %d)"
          % ("ESTABLE" if stable else "INESTABLE", MAX_RANGE_Y, MAX_RANGE_CRCB))


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolucion real: %dx%d" % (w, h))
    show_props(cap, "valores iniciales")

    times = []
    recording = False
    t_start = 0.0
    samples = []
    frame_count = 0
    real = (cap.get(cv2.CAP_PROP_AUTO_WB), cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            cap.get(cv2.CAP_PROP_EXPOSURE))
    px, py, pw, ph = PATCH

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break
        frame_count = frame_count + 1

        # FPS sobre los ultimos N_FPS frames
        times.append(time.time())
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        # Estado real del driver (se refresca cada 15 frames)
        if frame_count % 15 == 0:
            real = (cap.get(cv2.CAP_PROP_AUTO_WB), cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
                    cap.get(cv2.CAP_PROP_EXPOSURE))

        # Valores del parche de pared y saturacion de toda la imagen
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        patch = ycc[py:py + ph, px:px + pw].reshape(-1, 3)
        y_p, cr_p, cb_p = patch[:, 0].mean(), patch[:, 1].mean(), patch[:, 2].mean()
        saturated = 100.0 * np.mean(ycc[:, :, 0] >= 250)
        if recording:
            samples.append((y_p, cr_p, cb_p))
            if time.time() - t_start >= MEASURE_SECONDS:
                recording = False
                report_stability(samples)

        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %.1f   Saturado: %.1f%%" % (fps, saturated), (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "Parche: Y %.0f  Cr %.0f  Cb %.0f" % (y_p, cr_p, cb_p), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "Driver: WB_auto %.0f  T %.0f  Expo %.0f" % real, (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if recording:
            left = MEASURE_SECONDS - (time.time() - t_start)
            cv2.putText(frame, "MIDIENDO... %.0f s (mueve la mano)" % left, (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("ajustes de camara", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            try_set(cap, "AUTO_WB", cv2.CAP_PROP_AUTO_WB, 0)
        elif key == ord("a"):
            try_set(cap, "AUTO_WB", cv2.CAP_PROP_AUTO_WB, 1)
        elif key == ord("+") or key == ord("-"):
            step = 200 if key == ord("+") else -200
            try_set(cap, "WB_TEMPERATURE", cv2.CAP_PROP_WB_TEMPERATURE,
                    cap.get(cv2.CAP_PROP_WB_TEMPERATURE) + step)
        elif key == ord("x"):
            try_set(cap, "AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE, EXPO_MANUAL)
        elif key == ord("z"):
            try_set(cap, "AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE, EXPO_AUTO)
        elif key == ord("e") or key == ord("d"):
            step = 1 if key == ord("e") else -1
            try_set(cap, "EXPOSURE", cv2.CAP_PROP_EXPOSURE,
                    cap.get(cv2.CAP_PROP_EXPOSURE) + step)
        elif key == ord("l"):
            if not recording:
                recording = True
                samples = []
                t_start = time.time()
                print("\nMidiendo %d s... mueve la mano delante de la camara (sin tapar el parche verde)"
                      % MEASURE_SECONDS)
        elif key == ord("p"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)      # ventana de ajustes del driver

    show_props(cap, "valores finales (apuntalos)")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
