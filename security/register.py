# register.py - Registro del dataset de letras C, U, B, I, L (Fase 3)
#
# Graba 40 frames por letra (4 bloques de 10: centrada, mas/menos antebrazo, giro a un
# lado, giro al otro). Mover la mano sin girarla no cambia la mascara 64x64 (el recorte
# se centra en la mano), por eso se pide variar giro y antebrazo visible con el mismo pipeline de segmentation.py, y los guarda en
# security/data/dataset.npz junto con sus features (Hu, geometria, HOG).
#
# Uso:  python register.py                 graba en data/dataset.npz (si existe, continua)
#       python register.py --out test_x.npz   otro fichero (p. ej. grabacion de otro dia / luz)
#       python register.py --resumen       sin camara: tabla de features y montaje del fichero
#
# Orden al arrancar (la pantalla indica siempre el siguiente paso):
#   1) quita la mano de la ROI y pulsa b       -> fondo de referencia
#   2) palma sobre el cuadrado cian y pulsa p  -> color de tu piel
#   3) elige letra con 1 C | 2 U | 3 B | 4 I | 5 L  y pulsa ESPACIO para grabar
#
# Teclas: b, p     fondo y piel (como en segmentation.py)
#         1-5      elige la letra (C U B I L)
#         ESPACIO  empieza / pausa la grabacion de esa letra
#         x        borra los frames de la letra elegida (para regrabarla)
#         t / g    umbral de la resta de fondo +/-   (solo sin grabar)
#         r / f    margen de la piel +/-             (solo sin grabar)
#         q        guardar y salir

import argparse
import os
import time
import cv2
import numpy as np

import segmentation as seg
from cam import open_camera, driver_state
from camera import load_maps
from features import extract, GEO_NAMES

LETTERS = ["C", "U", "B", "I", "L"]
LETTER_KEYS = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3, ord("5"): 4}
N_PER_LETTER = 40
BLOCK_SIZE = 10
BLOCKS = ["centrada",
          "mas / menos antebrazo visible (mano mas dentro o mas fuera de la ROI)",
          "girada ~15 grados hacia la izquierda de la pantalla",
          "girada ~15 grados hacia la derecha de la pantalla"]

# lado de la ROI por el que entra el antebrazo (el unico donde la mano puede tocar el borde)
ENTRY = {"C": "abajo", "U": "abajo", "B": "derecha", "I": "abajo", "L": "abajo"}

EXPOSURE = -4              # la misma que se uso en segmentation.py
MARGIN = 2                 # px: tocar el borde = estar a <= MARGIN px de el
N_QUIET = 5                # frames seguidos con la mano quieta antes de guardar
IOU_QUIET = 0.85           # IoU minimo con el frame anterior para decir "quieta"
IOU_NEW = 0.95             # IoU maximo con el ultimo guardado (si es mayor, es el mismo)
MIN_GAP = 0.3              # segundos minimos entre dos frames guardados

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MONTAGE_PATH = os.path.join(HERE, "captures", "dataset_montaje.png")


# ------------------------------------------------------------ comprobaciones

def iou(a, b):
    """IoU de dos mascaras binarias."""
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return np.count_nonzero(a & b) / union


def border_problem(bbox, letter, w, h):
    """Devuelve el lado de la ROI que toca la mano (y no deberia), o None si esta bien."""
    x, y, bw, bh = bbox
    touches = {"izquierda": x <= MARGIN, "derecha": x + bw >= w - MARGIN,
               "arriba": y <= MARGIN, "abajo": y + bh >= h - MARGIN}
    for side, touched in touches.items():
        if touched and side != ENTRY[letter]:
            return side
    return None


# ------------------------------------------------------------ fichero .npz

def save(path, rows, roi, exposure):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        masks=np.array([r["mask"] for r in rows], np.uint8).reshape(-1, seg.OUT, seg.OUT),
        labels=np.array([r["label"] for r in rows]),
        hu=np.array([r["hu"] for r in rows], np.float32).reshape(-1, 7),
        geo=np.array([r["geo"] for r in rows], np.float32).reshape(-1, 4),
        hog=np.array([r["hog"] for r in rows], np.float32).reshape(len(rows), -1),
        bloque=np.array([r["bloque"] for r in rows], np.uint8),
        skin_lo=np.array([r["lo"] for r in rows], np.float32).reshape(-1, 2),
        skin_hi=np.array([r["hi"] for r in rows], np.float32).reshape(-1, 2),
        diff_thresh=np.array([r["diff"] for r in rows], np.int32),
        roi=np.array(roi), exposure=exposure)


def load(path):
    z = np.load(path)
    rows = []
    for i in range(len(z["labels"])):
        rows.append({"mask": z["masks"][i], "label": str(z["labels"][i]), "hu": z["hu"][i],
                     "geo": z["geo"][i], "hog": z["hog"][i], "bloque": int(z["bloque"][i]),
                     "lo": z["skin_lo"][i], "hi": z["skin_hi"][i], "diff": int(z["diff_thresh"][i])})
    return rows, [int(v) for v in z["roi"]], float(z["exposure"])


def count(rows, letter):
    return sum(1 for r in rows if r["label"] == letter)


# ------------------------------------------------------------ resumen y montaje

def make_montage(rows):
    """Una fila de 2x20 mascaras por letra, para revisar el dataset de un vistazo."""
    side, cols = seg.OUT, 20
    blocks = []
    for letter in LETTERS:
        masks = [r["mask"] for r in rows if r["label"] == letter]
        canvas = np.zeros((2 * side, cols * side), np.uint8)
        for i, m in enumerate(masks[:2 * cols]):
            canvas[(i // cols) * side:(i // cols + 1) * side, (i % cols) * side:(i % cols + 1) * side] = m
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, "%s (%d)" % (letter, len(masks)), (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 255), 2)
        blocks.append(canvas)
        blocks.append(np.full((4, cols * side, 3), 90, np.uint8))        # separador
    return np.vstack(blocks)


def print_summary(rows):
    print("\n%-3s %3s | %-23s | %s" % ("", "n", "geometria (media +- std)", "Hu log (media)"))
    print("    " + "    | " + "  ".join("%9s" % n for n in GEO_NAMES))
    for letter in LETTERS:
        sel = [r for r in rows if r["label"] == letter]
        if not sel:
            print("%-3s %3d | (sin datos)" % (letter, 0))
            continue
        geo = np.array([r["geo"] for r in sel])
        hu = np.array([r["hu"] for r in sel])
        print("%-3s %3d | " % (letter, len(sel)) +
              "  ".join("%4.2f+-%4.2f" % (m, s) for m, s in zip(geo.mean(0), geo.std(0))) +
              " | " + " ".join("%5.2f" % v for v in hu.mean(0)))


def show_summary(path):
    rows, roi, expo = load(path)
    print("%s: %d muestras | ROI %s | exposicion %.0f" % (path, len(rows), roi, expo))
    print_summary(rows)
    cv2.imwrite(MONTAGE_PATH, make_montage(rows))
    print("Montaje guardado en %s" % MONTAGE_PATH)


# ------------------------------------------------------------ programa principal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset.npz", help="nombre del fichero dentro de data/")
    parser.add_argument("--resumen", action="store_true", help="solo resumen y montaje (sin camara)")
    args = parser.parse_args()
    path = os.path.join(DATA_DIR, args.out)

    if args.resumen:
        show_summary(path)
        return

    roi = list(seg.ROI)
    rows = []
    if os.path.exists(path):
        rows, roi_old, expo_old = load(path)
        if roi_old != roi or expo_old != EXPOSURE:
            print("ERROR: %s se grabo con ROI %s y exposicion %.0f; ahora son %s y %d. "
                  "Usa otro --out o borra el fichero." % (path, roi_old, expo_old, roi, EXPOSURE))
            return
        print("Cargadas %d muestras de %s" % (len(rows), path))

    size, map1, map2 = load_maps()
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    expo_read = cap.get(cv2.CAP_PROP_EXPOSURE)
    print("Driver:", driver_state(cap))
    if expo_read != EXPOSURE:
        print("AVISO: la exposicion leida es %.0f y deberia ser %d" % (expo_read, EXPOSURE))

    bg = None
    bg_chroma = None
    skin_center = None
    skin_std = None
    half_min = seg.MIN_HALF
    diff_thresh = seg.DIFF_THRESH
    collecting = None
    bg_frames = []
    skin_frames = []

    current = None               # indice de la letra elegida (None = ninguna)
    recording = False
    prev_mask = None             # mascara del frame anterior (para saber si esta quieta)
    last_saved = None            # ultima mascara guardada
    last_time = 0.0
    quiet = 0                    # frames seguidos con la mano quieta
    msg = ""

    cv2.namedWindow("registro")
    cv2.moveWindow("registro", 0, 0)
    cv2.namedWindow("mascara")
    cv2.moveWindow("mascara", 660, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        roi_img = seg.get_roi(frame, roi)
        h_roi, w_roi = roi_img.shape[:2]

        # captura de frames para el fondo o la piel (igual que en segmentation.py)
        if collecting == "bg":
            bg_frames.append(roi_img.astype(np.float32))
            if len(bg_frames) == seg.N_BG:
                bg = cv2.GaussianBlur(np.mean(bg_frames, axis=0).astype(np.uint8), (5, 5), 0)
                bg_chroma = seg.chroma_of(bg)
                collecting = None
                print("Fondo capturado: Cr = %.1f | Cb = %.1f" % (bg_chroma[0], bg_chroma[1]))
        elif collecting == "skin":
            skin_frames.append(seg.skin_samples(roi_img))
            if len(skin_frames) == seg.N_SKIN:
                data = np.vstack(skin_frames)
                skin_center = data.mean(axis=0)
                skin_std = data.std(axis=0)
                collecting = None
                print("Piel: Cr = %.1f (std %.1f) | Cb = %.1f (std %.1f)"
                      % (skin_center[0], skin_std[0], skin_center[1], skin_std[1]))
                if skin_std.max() > seg.MAX_STD:
                    print("AVISO: la muestra no es uniforme (std > %d). Repite con p." % seg.MAX_STD)

        # segmentacion (solo con fondo y piel listos)
        pasos, info, lo, hi = None, None, None, None
        wall_in_skin = None
        if bg is not None and skin_center is not None and collecting is None:
            lo, hi = seg.skin_range(skin_center, skin_std, half_min)
            wall_in_skin = bool(np.all((bg_chroma >= lo) & (bg_chroma <= hi)))
            pasos, info = seg.segment(roi_img, bg, lo, hi, diff_thresh)

        mask = pasos["mascara64"] if info is not None else None

        # ---- estado de la mano: quieta o no
        if mask is None:
            quiet = 0
            prev_mask = None
        else:
            if prev_mask is not None and iou(mask > 0, prev_mask > 0) >= IOU_QUIET:
                quiet += 1
            else:
                quiet = 0
            prev_mask = mask

        # ---- decidir si este frame es valido y se guarda
        letter = LETTERS[current] if current is not None else None
        n_done = count(rows, letter) if letter else 0
        block = min(n_done // BLOCK_SIZE, len(BLOCKS) - 1)
        reason = ""
        if collecting is not None or bg is None or skin_center is None:
            reason = ""
        elif wall_in_skin:
            reason = "el fondo cae en el rango de piel (mas luz, o f)"
        elif letter is None:
            reason = "elige letra con 1-5"
        elif not recording:
            reason = "pausado: ESPACIO para grabar"
        elif info is None:
            reason = "no hay mano"
        else:
            side = border_problem(info["bbox"], letter, w_roi, h_roi)
            if side:
                reason = "la mano toca el borde %s de la ROI" % side
            elif quiet < N_QUIET:
                reason = "mano quieta... (%d/%d)" % (quiet, N_QUIET)
            elif last_saved is not None and iou(mask > 0, last_saved > 0) >= IOU_NEW:
                reason = "igual que el ultimo: muevete un poco"
            elif time.time() - last_time < MIN_GAP:
                reason = "esperando..."
            else:
                feats = extract(mask)
                if feats is None:
                    reason = "mascara vacia"
                else:
                    hu, geo, hog = feats
                    rows.append({"mask": mask.copy(), "label": letter, "hu": hu, "geo": geo,
                                 "hog": hog, "bloque": block, "lo": lo.copy(), "hi": hi.copy(),
                                 "diff": diff_thresh})
                    last_saved = mask.copy()
                    last_time = time.time()
                    quiet = 0
                    n_done += 1
                    reason = "GUARDADO"
                    if n_done >= N_PER_LETTER:
                        recording = False
                        save(path, rows, roi, EXPOSURE)
                        msg = "%s completa (%d frames) y guardada" % (letter, n_done)
                        print(msg)
                        last_saved = None

        # ---- pantalla
        view = frame.copy()
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 255), 2)
        sx, sy, sw, sh = seg.sample_box(w, h)
        cv2.rectangle(view, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 255, 0), 1)
        counts = "  ".join("%s %d/%d" % (l, count(rows, l), N_PER_LETTER) for l in LETTERS)
        seg.put(view, counts, 25, (0, 255, 0), 0.6)
        if collecting == "bg":
            hint = "Capturando fondo (%d/%d): no muevas nada" % (len(bg_frames), seg.N_BG)
        elif collecting == "skin":
            hint = "Capturando piel (%d/%d): palma quieta" % (len(skin_frames), seg.N_SKIN)
        elif bg is None:
            hint = "PASO 1: quita la mano de la ROI y pulsa b"
        elif skin_center is None:
            hint = "PASO 2: palma sobre el cuadrado cian y pulsa p"
        elif letter is None:
            hint = "PASO 3: elige letra con 1 C | 2 U | 3 B | 4 I | 5 L"
        else:
            hint = "%s (entra por %s) | bloque %d/4: %s" % (letter, ENTRY[letter], block + 1, BLOCKS[block])
        seg.put(view, hint, 50, (0, 220, 255), 0.5)
        color = (0, 255, 0) if reason == "GUARDADO" else (0, 165, 255)
        seg.put(view, ("REC  " if recording else "") + reason, 75, color, 0.6)
        if info is not None:
            feats_now = extract(mask)
            if feats_now is not None:
                g = feats_now[1]
                seg.put(view, "sol %.2f asp %.2f ext %.2f def %d | area %d"
                        % (g[0], g[1], g[2], g[3], info["area"]), 100, (255, 255, 255), 0.5)
        if msg:
            seg.put(view, msg, 125, (255, 255, 0), 0.5)
        cv2.imshow("registro", view)

        small = np.zeros((seg.OUT, seg.OUT), np.uint8) if mask is None else mask
        cv2.imshow("mascara", cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b") and not recording:
            bg_frames = []
            collecting = "bg"
        elif key == ord("p") and not recording:
            skin_frames = []
            collecting = "skin"
        elif key in LETTER_KEYS:
            current = LETTER_KEYS[key]
            recording = False
            last_saved = None
            msg = ""
        elif key == ord(" "):
            if current is None:
                print("Elige primero la letra con 1-5")
            elif bg is None or skin_center is None:
                print("Faltan el fondo (b) y la piel (p)")
            elif count(rows, LETTERS[current]) >= N_PER_LETTER:
                print("%s ya esta completa: pulsa x para regrabarla" % LETTERS[current])
            else:
                recording = not recording
                last_saved = None
                quiet = 0
        elif key == ord("x") and current is not None:
            rows = [r for r in rows if r["label"] != LETTERS[current]]
            recording = False
            last_saved = None
            msg = "%s borrada" % LETTERS[current]
            print(msg)
        elif key == ord("t") and not recording:
            diff_thresh = min(diff_thresh + seg.DIFF_STEP, 255)
        elif key == ord("g") and not recording:
            diff_thresh = max(diff_thresh - seg.DIFF_STEP, 0)
        elif key == ord("r") and not recording:
            half_min += seg.HALF_STEP
        elif key == ord("f") and not recording:
            half_min = max(half_min - seg.HALF_STEP, 0)

    cap.release()
    cv2.destroyAllWindows()
    if rows:
        save(path, rows, roi, EXPOSURE)
        print("Guardadas %d muestras en %s" % (len(rows), path))
        print_summary(rows)
        cv2.imwrite(MONTAGE_PATH, make_montage(rows))
        print("Montaje en %s" % MONTAGE_PATH)
    else:
        print("No hay muestras: no se guarda nada")


if __name__ == "__main__":
    main()
