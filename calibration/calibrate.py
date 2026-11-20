# calibrate.py - Calibracion offline de la camara (Fase 1, paso 5)
#
# Uso:  python calibrate.py
# Lee calibration/images/img_*.png y guarda calibration/calib.npz con:
#   K (matriz intrinseca), dist (distorsion), rms, img_size (ancho, alto),
#   square_mm, pattern

import glob
import os
import cv2
import numpy as np

PATTERN = (6, 8)            # esquinas interiores del tablero impreso (7x9 cuadros)
SQUARE_MM = 25.0            # lado del cuadro, medido con regla
WIN = 5                     # ventana de cornerSubPix: tiene que ser menor que la mitad
                            # del cuadro en pixeles (aqui ~12 px). Con 11 las esquinas se corrompen
FLAGS_DETECT = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
# Modelo de distorsion: solo k1 y k2 (radial). La tangencial y k3 salen sin significado
# estadistico con estas imagenes y empeoran la estabilidad de K.
FLAGS_CALIB = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images")
OUT_FILE = os.path.join(HERE, "calib.npz")


def main():
    # Puntos 3D del tablero (plano Z = 0), en mm
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = []            # puntos 3D de cada imagen
    img_points = []            # esquinas detectadas en cada imagen
    used = []
    img_size = None

    files = sorted(glob.glob(os.path.join(IMAGES_DIR, "img_*.png")))
    print("Imagenes encontradas: %d" % len(files))
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])        # (ancho, alto)
        elif (gray.shape[1], gray.shape[0]) != img_size:
            print("  %s: tamano distinto, se ignora" % os.path.basename(f))
            continue

        found, corners = cv2.findChessboardCorners(gray, PATTERN, FLAGS_DETECT)
        if not found:
            print("  %s: tablero NO detectado, se ignora" % os.path.basename(f))
            continue
        corners = cv2.cornerSubPix(gray, corners, (WIN, WIN), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(os.path.basename(f))

    if len(used) < 10:
        print("Hay menos de 10 imagenes validas: no se calibra")
        return
    print("Imagenes usadas: %d | resolucion: %dx%d" % (len(used), img_size[0], img_size[1]))

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=FLAGS_CALIB)

    print("\nRMS (error de reproyeccion): %.3f px" % rms)
    print("K =\n", np.round(K, 2))
    print("dist (k1, k2, p1, p2, k3) =", np.round(dist.ravel(), 4))

    # Error de reproyeccion de cada imagen (para detectar y descartar las malas)
    print("\nError por imagen (px):")
    for i in range(len(used)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        diff = img_points[i].reshape(-1, 2) - proj.reshape(-1, 2)
        err = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))      # RMS de la imagen
        print("  %s: %.3f" % (used[i], err))

    np.savez(OUT_FILE, K=K, dist=dist, rms=rms, img_size=np.array(img_size),
             square_mm=SQUARE_MM, pattern=np.array(PATTERN))
    print("\nGuardado en", OUT_FILE)


if __name__ == "__main__":
    main()
