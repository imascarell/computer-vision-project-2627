# segmentation.py - Segmentacion de la mano (Fase 2)
#
# Pipeline sobre la ROI:
#   ROI -> YCrCb -> piel (umbral en Cr y Cb) -> resta del fondo de referencia
#       -> AND de ambas -> morfologia (apertura + cierre) -> contorno mayor relleno
#       -> recorte cuadrado -> mascara binaria 64x64
#
# Uso:  python segmentation.py
# Orden al arrancar (la pantalla indica siempre el siguiente paso):
#   1) quita la mano de la ROI y pulsa b      -> guarda el fondo de referencia
#   2) pon la palma sobre el cuadrado cian y pulsa p -> calibra el color de tu piel
#   3) pon un signo en la ROI y pulsa s       -> guarda las imagenes de cada paso
#
# Teclas: b     -> capturar el fondo (ROI sin mano)
#         p     -> calibrar la piel (palma sobre el cuadrado cian)
#         s     -> guardar los pasos en security/captures/ (solo si hay mano)
#         i k j l -> mover la ROI (arriba, abajo, izquierda, derecha; 10 px)
#         t / g -> subir / bajar el umbral de la resta de fondo
#         r / f -> ampliar / estrechar el margen del umbral de piel
#         e / d -> subir / bajar la exposicion (invalida fondo y piel)
#         q     -> salir (imprime los valores finales para copiarlos a las constantes)

import glob
import os
import time
import cv2
import numpy as np

from cam import open_camera, driver_state
from camera import load_maps

# ROI: x, y, ancho, alto. Zona donde se coloca la mano derecha (lado derecho de la imagen),
# fuera de la cara. Se puede mover con i k j l y copiar el valor final aqui.
ROI = [360, 120, 240, 240]
STEP = 10                  # pixeles que mueve la ROI cada pulsacion
N_FPS = 30                 # frames para los FPS en pantalla
N_BG = 30                  # frames que se promedian para el fondo
N_SKIN = 15                # frames que se usan para calibrar la piel
SAMPLE = 60                # lado (px) del cuadrado donde se pone la palma
MIN_HALF = 8               # margen minimo (en niveles de Cr y Cb) alrededor del color de la piel
HALF_STEP = 2              # cuanto cambian r / f ese margen
MAX_STD = 12               # si la desviacion de la muestra de piel supera esto, se avisa
DIFF_THRESH = 25           # diferencia minima con el fondo (0-255) para contar como "algo nuevo"
DIFF_STEP = 5              # cuanto cambian t / g ese umbral
KERNEL = 5                 # tamano del elemento estructurante de la morfologia
MIN_AREA = 1500            # area minima (px) del contorno mayor para aceptarlo como mano
OUT = 64                   # lado de la mascara final
PAD = 1.1                  # el cuadrado de recorte es un 10 % mayor que la mano (borde negro)

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(HERE, "captures")
STEP_NAMES = ["roi", "piel", "resta", "and", "morfologia", "contorno", "mascara64"]


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


# ---------------------------------------------------------------- piel

def sample_box(roi_w, roi_h):
    """Cuadrado central de la ROI donde se pone la palma para calibrar la piel."""
    return (roi_w - SAMPLE) // 2, (roi_h - SAMPLE) // 2, SAMPLE, SAMPLE


def skin_samples(roi_img):
    """Cr y Cb de los pixeles del cuadrado de muestra. Devuelve un array de N x 2."""
    x, y, w, h = sample_box(roi_img.shape[1], roi_img.shape[0])
    ycrcb = cv2.cvtColor(roi_img[y:y + h, x:x + w], cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 1:3].reshape(-1, 2)


def chroma_of(img):
    """Cr y Cb tipicos (mediana) de una imagen BGR; sirve para comparar el fondo con la piel."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return np.median(ycrcb[:, :, 1:3].reshape(-1, 2), axis=0)


def skin_range(center, std, half_min):
    """Rango de Cr y Cb: centro +- max(3 desviaciones, margen minimo)."""
    half = np.maximum(3 * std, half_min)
    lo = np.clip(center - half, 0, 255).astype(int)
    hi = np.clip(center + half, 0, 255).astype(int)
    return lo, hi


# ------------------------------------------------------------ segmentacion

def square_crop(mask, bbox):
    """Recorta un cuadrado centrado en la bounding box y lo lleva a OUT x OUT (binario)."""
    x, y, w, h = bbox
    side = int(max(w, h) * PAD)
    x0 = x + w // 2 - side // 2
    y0 = y + h // 2 - side // 2
    # se rellena con negro por si el cuadrado se sale de la ROI
    padded = cv2.copyMakeBorder(mask, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    crop = padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    small = cv2.resize(crop, (OUT, OUT), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    return small


def segment(roi_img, bg, lo, hi, diff_thresh):
    """Aplica el pipeline a la ROI. bg = fondo ya suavizado (mismo tamano que la ROI).

    Devuelve (pasos, info). pasos es un diccionario con una imagen por paso (claves de
    STEP_NAMES). info es None si no hay mano, o un diccionario con el area y la bounding box.
    """
    # 1) piel: umbral solo en Cr y Cb (la luminancia Y queda libre, asi la sombra no la rompe)
    ycrcb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, int(lo[0]), int(lo[1])), (255, int(hi[0]), int(hi[1])))

    # 2) resta del fondo: cambio maximo entre canales, tras suavizar el ruido
    blur = cv2.GaussianBlur(roi_img, (5, 5), 0)
    diff = cv2.absdiff(blur, bg).max(axis=2)
    resta = np.where(diff > diff_thresh, 255, 0).astype(np.uint8)

    # 3) AND: solo es mano lo que tiene color de piel Y no estaba en el fondo
    both = cv2.bitwise_and(skin, resta)

    # 4) morfologia: apertura (quita puntos sueltos) y cierre (tapa agujeros pequenos)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
    clean = cv2.morphologyEx(both, cv2.MORPH_OPEN, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)

    # 5) contorno mayor, relleno
    contour_vis = roi_img.copy()
    mask64 = np.zeros((OUT, OUT), np.uint8)
    info = None
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area >= MIN_AREA:
            filled = np.zeros_like(clean)
            cv2.drawContours(filled, [biggest], -1, 255, cv2.FILLED)
            bbox = cv2.boundingRect(biggest)
            mask64 = square_crop(filled, bbox)
            info = {"area": area, "bbox": bbox}
            x, y, w, h = bbox
            cv2.drawContours(contour_vis, [biggest], -1, (0, 255, 0), 2)
            cv2.rectangle(contour_vis, (x, y), (x + w, y + h), (0, 220, 255), 1)

    pasos = {"roi": roi_img, "piel": skin, "resta": resta, "and": both,
             "morfologia": clean, "contorno": contour_vis, "mascara64": mask64}
    return pasos, info


# ---------------------------------------------------------------- pantalla

def make_mosaic(pasos, size):
    """Une los pasos en una sola imagen (4 columnas x 2 filas) para depurar de un vistazo."""
    w, h = size
    labels = ["1 ROI", "2 Piel Cr/Cb", "3 Resta fondo", "4 Piel AND resta",
              "5 Morfologia", "6 Contorno mayor", "7 Mascara 64x64"]
    tiles = []
    for name, label in zip(STEP_NAMES, labels):
        img = pasos[name]
        if name == "mascara64":
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.copy()
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        tiles.append(img)
    tiles.append(np.zeros((h, w, 3), np.uint8))               # hueco para completar la rejilla
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


def save_steps(pasos):
    """Guarda una imagen por paso en captures/ (para la secuencia de transformaciones del informe)."""
    n = len(glob.glob(os.path.join(CAPTURES_DIR, "seg_*_01_roi.png"))) + 1
    for i, name in enumerate(STEP_NAMES, start=1):
        cv2.imwrite(os.path.join(CAPTURES_DIR, "seg_%02d_%02d_%s.png" % (n, i, name)), pasos[name])
    print("Guardadas %d imagenes: %s" % (len(STEP_NAMES),
          os.path.join(CAPTURES_DIR, "seg_%02d_*.png" % n)))


def put(view, text, y, color=(0, 255, 0), scale=0.6):
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    size, map1, map2 = load_maps()
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    print("Driver:", driver_state(cap))

    roi = list(ROI)
    bg = None                    # fondo de referencia suavizado (None = falta)
    bg_chroma = None             # Cr, Cb tipicos del fondo
    skin_center = None           # Cr, Cb medios de la piel (None = falta)
    skin_std = None
    half_min = MIN_HALF
    diff_thresh = DIFF_THRESH
    collecting = None            # "bg" o "skin" mientras se estan capturando frames
    bg_frames = []
    skin_frames = []
    pasos, info = None, None
    times = []

    cv2.namedWindow("segmentacion")
    cv2.moveWindow("segmentacion", 0, 0)
    cv2.namedWindow("pasos")
    cv2.moveWindow("pasos", 660, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)     # undistort

        times.append(time.time())
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        roi_img = get_roi(frame, roi)

        # captura de frames para el fondo o para la piel
        if collecting == "bg":
            bg_frames.append(roi_img.astype(np.float32))
            if len(bg_frames) == N_BG:
                mean = np.mean(bg_frames, axis=0).astype(np.uint8)
                bg = cv2.GaussianBlur(mean, (5, 5), 0)
                bg_chroma = chroma_of(bg)
                print("Fondo: Cr = %.1f | Cb = %.1f" % (bg_chroma[0], bg_chroma[1]))
                collecting = None
                print("Fondo capturado (%d frames)" % N_BG)
        elif collecting == "skin":
            skin_frames.append(skin_samples(roi_img))
            if len(skin_frames) == N_SKIN:
                data = np.vstack(skin_frames)
                skin_center = data.mean(axis=0)
                skin_std = data.std(axis=0)
                collecting = None
                print("Piel: Cr = %.1f (std %.1f) | Cb = %.1f (std %.1f)"
                      % (skin_center[0], skin_std[0], skin_center[1], skin_std[1]))
                if skin_std.max() > MAX_STD:
                    print("AVISO: la muestra no es uniforme (std > %d). "
                          "Cubre bien el cuadrado cian con la palma y repite con p." % MAX_STD)

        # segmentacion (solo con fondo y piel listos)
        pasos, info = None, None
        wall_in_skin = None
        if bg is not None and skin_center is not None and collecting is None:
            lo, hi = skin_range(skin_center, skin_std, half_min)
            # si el color del fondo cae dentro del rango de piel, la piel no separa nada
            wall_in_skin = bool(np.all((bg_chroma >= lo) & (bg_chroma <= hi)))
            pasos, info = segment(roi_img, bg, lo, hi, diff_thresh)

        # ventana principal
        view = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 255), 2)
        sx, sy, sw, sh = sample_box(w, h)
        cv2.rectangle(view, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 255, 0), 1)
        put(view, "FPS: %.1f   Expo: %.0f" % (fps, cap.get(cv2.CAP_PROP_EXPOSURE)), 25, scale=0.7)
        put(view, "Fondo: %s   Piel: %s   Umbral resta: %d   Margen piel: %d"
            % ("OK" if bg is not None else "FALTA", "OK" if skin_center is not None else "FALTA",
               diff_thresh, half_min), 50, (0, 255, 0), 0.5)

        if collecting == "bg":
            hint = "Capturando fondo (%d/%d): no muevas nada" % (len(bg_frames), N_BG)
        elif collecting == "skin":
            hint = "Capturando piel (%d/%d): palma quieta sobre el cuadrado" % (len(skin_frames), N_SKIN)
        elif bg is None:
            hint = "PASO 1: quita la mano de la ROI y pulsa b"
        elif skin_center is None:
            hint = "PASO 2: palma sobre el cuadrado cian y pulsa p"
        elif info is None:
            hint = "Pon un signo dentro de la ROI"
        else:
            hint = "Mano detectada (area %d px): pulsa s para guardar" % info["area"]
        put(view, hint, 75, (0, 220, 255), 0.6)
        if wall_in_skin is not None:
            if wall_in_skin:
                put(view, "AVISO: el fondo cae en el rango de piel (mas luz, o pulsa f)", 100, (0, 0, 255), 0.5)
            else:
                put(view, "Fondo fuera del rango de piel: OK", 100, (0, 255, 0), 0.5)
        cv2.imshow("segmentacion", view)

        if pasos is not None:
            cv2.imshow("pasos", make_mosaic(pasos, (w, h)))
        else:
            cv2.imshow("pasos", make_mosaic({"roi": roi_img, "piel": np.zeros((h, w), np.uint8),
                                             "resta": np.zeros((h, w), np.uint8),
                                             "and": np.zeros((h, w), np.uint8),
                                             "morfologia": np.zeros((h, w), np.uint8),
                                             "contorno": roi_img,
                                             "mascara64": np.zeros((OUT, OUT), np.uint8)}, (w, h)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            bg_frames = []
            collecting = "bg"
        elif key == ord("p"):
            skin_frames = []
            collecting = "skin"
        elif key == ord("s"):
            if info is None:
                print("No hay mano detectada: no se guarda nada")
            else:
                save_steps(pasos)
        elif key in (ord("i"), ord("k"), ord("j"), ord("l")):
            dx = {ord("j"): -STEP, ord("l"): STEP}.get(key, 0)
            dy = {ord("i"): -STEP, ord("k"): STEP}.get(key, 0)
            roi = move_roi(roi, dx, dy, frame.shape[1], frame.shape[0])
            bg = None                       # el fondo era de la zona anterior
            collecting = None
            print("ROI = %s  (hay que repetir b)" % roi)
        elif key == ord("t"):
            diff_thresh = min(diff_thresh + DIFF_STEP, 255)
        elif key == ord("g"):
            diff_thresh = max(diff_thresh - DIFF_STEP, 0)
        elif key == ord("r"):
            half_min += HALF_STEP
        elif key == ord("f"):
            half_min = max(half_min - HALF_STEP, 0)
        elif key == ord("e") or key == ord("d"):
            new = cap.get(cv2.CAP_PROP_EXPOSURE) + (1 if key == ord("e") else -1)
            cap.set(cv2.CAP_PROP_EXPOSURE, new)
            print("Exposicion pedida %.0f | leida %.0f  (hay que repetir b y p)"
                  % (new, cap.get(cv2.CAP_PROP_EXPOSURE)))
            bg = None                       # con otra exposicion cambian fondo y color
            skin_center = None
            collecting = None

    cap.release()
    cv2.destroyAllWindows()
    print("ROI = %s" % roi)
    print("DIFF_THRESH = %d | MIN_HALF = %d" % (diff_thresh, half_min))
    if skin_center is not None:
        print("Piel: Cr = %.1f (std %.1f) | Cb = %.1f (std %.1f)"
              % (skin_center[0], skin_std[0], skin_center[1], skin_std[1]))


if __name__ == "__main__":
    main()
