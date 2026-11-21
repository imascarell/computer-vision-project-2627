# cam.py - Apertura de la camara con los ajustes fijos del proyecto (Fase 2)
#
# Todos los scripts abren la camara con open_camera(). Asi el balance de blancos y la
# exposicion no cambian solos (con el automatico el color del fondo llegaba a moverse
# ~20 puntos en Cr al entrar la mano; con estos ajustes, ~2).
#
# Nota: AUTO_EXPOSURE no se puede leer en esta camara (devuelve -1). La exposicion manual
# se fijo ademas desde el panel del driver (camera_settings.py, tecla 'p').

import os
import cv2

CAM_INDEX = 0
WIDTH = 640                # misma resolucion que en la calibracion
HEIGHT = 480
WB_TEMPERATURE = 4000      # balance de blancos manual (K)
EXPOSURE = -4              # exposicion manual (log2 de segundos: -4 = 1/16 s)


def open_camera():
    if os.name == "nt":
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)       # Windows (DirectShow)
        expo_manual = 0.25
    else:
        cap = cv2.VideoCapture(CAM_INDEX)
        expo_manual = 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, expo_manual)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    return cap


def driver_state(cap):
    """Texto con el estado real del driver (para comprobarlo en pantalla o terminal)."""
    return "WB_auto %.0f | T %.0f | Expo %.0f" % (
        cap.get(cv2.CAP_PROP_AUTO_WB), cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        cap.get(cv2.CAP_PROP_EXPOSURE))
