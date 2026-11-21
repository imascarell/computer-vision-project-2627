# features.py - Features de una mascara binaria 64x64 (Fase 3)
#
# extract(mask) devuelve (hu, geo, hog):
#   hu  : 7 momentos de Hu en escala log (invariantes a traslacion, escala y giro)
#   geo : [solidez, relacion de aspecto, extension, n defectos de convexidad]
#   hog : HOG de la mascara (1764 valores)
#
# La misma funcion se usa para grabar el dataset y (mas adelante) en la demo.
#
# Prueba sin camara:  python features.py              (usa las 5 capturas elegidas)
#                     python features.py a.png b.png  (usa las mascaras que le pases)

import os
import sys
import cv2
import numpy as np

OUT = 64                   # lado de la mascara
MIN_DEPTH = 3.0            # profundidad minima (px de la mascara 64x64) de un defecto de convexidad
GEO_NAMES = ["solidez", "aspecto", "extension", "defectos"]

# HOG: ventana 64x64, bloque 16x16, paso 8, celda 8, 9 orientaciones
# -> 7 x 7 bloques x 4 celdas x 9 = 1764 valores
_hog = cv2.HOGDescriptor((OUT, OUT), (16, 16), (8, 8), (8, 8), 9)

HERE = os.path.dirname(os.path.abspath(__file__))
# captura -> letra (las 5 elegidas en la Fase 2)
DEFAULT_CAPTURES = [("C", 1), ("U", 7), ("B", 6), ("I", 4), ("L", 5)]


def biggest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def hu_log(mask):
    """7 momentos de Hu en log10 con signo (si no, los ultimos valen ~1e-20)."""
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def geometry(contour):
    """[solidez, aspecto, extension, defectos] a partir del contorno de la mano."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    solidity = area / cv2.contourArea(hull)

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h                  # ancho / alto del contorno
    extent = area / (w * h)         # area / area de la bounding box

    # defectos de convexidad: huecos entre la mano y su envolvente (p. ej. entre dedos)
    n_defects = 0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:               # contornos raros: se cuentan 0 defectos
        defects = None
    if defects is not None:
        depth = defects.reshape(-1, 4)[:, 3] / 256.0   # OpenCV da la profundidad x256
        n_defects = int(np.sum(depth >= MIN_DEPTH))
    return np.array([solidity, aspect, extent, n_defects], np.float32)


def extract(mask):
    """mask: uint8 64x64 con valores 0 / 255. Devuelve (hu, geo, hog) o None si esta vacia."""
    contour = biggest_contour(mask)
    if contour is None or cv2.contourArea(contour) == 0:
        return None
    hu = hu_log(mask).astype(np.float32)
    geo = geometry(contour)
    hog = _hog.compute(mask).ravel().astype(np.float32)
    return hu, geo, hog


def main():
    if len(sys.argv) > 1:
        items = [(os.path.basename(p), p) for p in sys.argv[1:]]
    else:
        folder = os.path.join(HERE, "captures")
        items = [(l, os.path.join(folder, "seg_%02d_07_mascara64.png" % n))
                 for l, n in DEFAULT_CAPTURES]
    print("%-10s %6s | %s | %s" % ("mascara", "px", " ".join("%9s" % n for n in GEO_NAMES),
                                   " ".join("   hu%d" % i for i in range(1, 8))))
    for name, path in items:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("%-10s no se pudo leer %s" % (name, path))
            continue
        res = extract(mask)
        if res is None:
            print("%-10s mascara vacia" % name)
            continue
        hu, geo, hog = res
        print("%-10s %6d | %9.3f %9.3f %9.3f %9d | %s   (HOG: %d valores)"
              % (name, np.count_nonzero(mask), geo[0], geo[1], geo[2], geo[3],
                 " ".join("%5.2f" % v for v in hu), len(hog)))


if __name__ == "__main__":
    main()
