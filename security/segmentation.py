# segmentation.py - Segmentacion de la mano (Fase 2)
#
# Paso 2 (este archivo ahora): base = camara con ajustes fijos + undistort + ROI + FPS.
# Los pasos siguientes (fondo, piel, morfologia, contorno, 64x64) se anaden aqui.
#
# Uso:  python segmentation.py
# Teclas: w a s d -> mover la ROI (arriba, izquierda, abajo, derecha; 10 px por pulsacion)
#         q       -> salir (imprime la ROI para copiarla a las constantes)

import time
import cv2

from cam import open_camera, driver_state
from camera import load_maps

# ROI: x, y, ancho, alto. Zona donde se coloca la mano derecha (lado derecho de la imagen),
# fuera de la cara. Se puede mover con w a s d y copiar el valor final aqui.
ROI = [360, 120, 240, 240]
STEP = 10
N_FPS = 30


def move_roi(roi, dx, dy, frame_w, frame_h):
    """Mueve la ROI sin salirse de la imagen."""
    x, y, w, h = roi
    x = min(max(x + dx, 0), frame_w - w)
    y = min(max(y + dy, 0), frame_h - h)
    return [x, y, w, h]


def get_roi(frame, roi):
    """Recorta la ROI del frame (aqui se procesara la mano)."""
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def main():
    size, map1, map2 = load_maps()
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    print("Driver:", driver_state(cap))

    roi = list(ROI)
    times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)     # undistort

        # FPS sobre los ultimos N_FPS frames
        times.append(time.time())
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        roi_img = get_roi(frame, roi)

        view = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(view, "ROI", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.putText(view, "FPS: %.1f" % fps, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("segmentacion", view)
        cv2.imshow("ROI", roi_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            roi = move_roi(roi, 0, -STEP, frame.shape[1], frame.shape[0])
        elif key == ord("s"):
            roi = move_roi(roi, 0, STEP, frame.shape[1], frame.shape[0])
        elif key == ord("a"):
            roi = move_roi(roi, -STEP, 0, frame.shape[1], frame.shape[0])
        elif key == ord("d"):
            roi = move_roi(roi, STEP, 0, frame.shape[1], frame.shape[0])

    cap.release()
    cv2.destroyAllWindows()
    print("ROI = %s" % roi)


if __name__ == "__main__":
    main()
