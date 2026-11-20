# distortion_map.py - Figura para el informe: cuanto mueve la correccion cada pixel
#
# Uso:  python distortion_map.py
# Lee calibration/calib.npz y guarda calibration/mapa_desplazamiento.png.
# No necesita camara.

import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "calib.npz")
OUT_FILE = os.path.join(HERE, "mapa_desplazamiento.png")


def main():
    data = np.load(CALIB_FILE)
    K = data["K"]
    dist = data["dist"]
    w, h = int(data["img_size"][0]), int(data["img_size"][1])

    # Con new_K = K se mide solo el efecto de la distorsion (sin zoom)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    disp = np.sqrt((map1 - xs) ** 2 + (map2 - ys) ** 2)       # desplazamiento en pixeles
    vmax = float(disp.max())
    print("Desplazamiento: maximo %.2f px | medio %.2f px | centro %.2f px"
          % (vmax, disp.mean(), disp[h // 2, w // 2]))

    # Mapa de color + barra de color a la derecha
    color = cv2.applyColorMap(np.uint8(255 * disp / vmax), cv2.COLORMAP_JET)
    bar_w = 90
    canvas = np.full((h, w + bar_w, 3), 255, np.uint8)
    canvas[:, :w] = color
    gradient = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
    canvas[:, w + 10:w + 30] = cv2.applyColorMap(np.repeat(gradient, 20, axis=1), cv2.COLORMAP_JET)
    for frac in (0.0, 0.5, 1.0):
        y = int((1 - frac) * (h - 1))
        text = "%.1f px" % (frac * vmax)
        cv2.putText(canvas, text, (w + 34, min(max(y + 5, 12), h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(OUT_FILE, canvas)
    print("Guardado en", OUT_FILE)


if __name__ == "__main__":
    main()
