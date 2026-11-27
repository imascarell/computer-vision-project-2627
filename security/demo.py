# demo.py - Sistema de seguridad en directo (Fase 5)
#
# Bucle: camara -> undistort -> ROI -> segmentacion -> features -> clasificador
#        -> decoder.py -> overlay.
#
# Uso:  python demo.py
# Orden al arrancar (la pantalla indica siempre el siguiente paso):
#   1) quita la mano de la ROI y pulsa b       -> fondo de referencia
#   2) palma sobre el cuadrado cian y pulsa p  -> color de tu piel
#   3) haz los 5 signos de la clave (por defecto C U B I L)
#
# Teclas:
#   b, p     fondo y piel (como en segmentation.py)
#   r        reset de la secuencia (tras ACCESO CONCEDIDO vuelve a pedir la clave;
#            durante el BLOQUEO no hace nada)
#   n        modo administrador: nueva clave (solo con ACCESO CONCEDIDO).
#            Haz los 5 signos; ENTER guarda, ESC cancela, r repite
#   q        salir (imprime los FPS medios)
#   Parametros del decodificador en vivo:
#   1 / 2    ventana N      -1 / +1 frames
#   3 / 4    ratio mayoria  -0.05 / +0.05
#   5 / 6    frames de neutro M  -1 / +1
#   7 / 8    timeout        -0.5 / +0.5 s
#   9 / 0    tau del clasificador  -0.02 / +0.02
#
# La clave nueva se guarda en data/clave.txt (si no existe, se usa CUBIL).

import os
import time
import cv2
import numpy as np

import decoder
import segmentation as seg
from cam import open_camera, driver_state
from camera import load_maps
from classifier import Classifier
from decoder import SequenceDecoder, draw_overlay
from features import extract

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "data", "modelo.npz")
KEY_FILE = os.path.join(HERE, "data", "clave.txt")
LETTERS = ["C", "U", "B", "I", "L"]
DEFAULT_KEY = ["C", "U", "B", "I", "L"]
KEY_LEN = 5
NOTE_TIME = 3.0            # segundos que se ve un aviso en pantalla
N_FPS = 30                 # frames para los FPS en pantalla


# ---------------------------------------------------------------- clave
def load_key(path):
    """Lee la clave de un fichero. Si no existe o no es valida, devuelve CUBIL."""
    if not os.path.exists(path):
        return list(DEFAULT_KEY)
    with open(path) as f:
        text = f.read().strip().upper()
    key = list(text)
    if len(key) != KEY_LEN or any(c not in LETTERS for c in key):
        print("AVISO: %s no tiene una clave valida: se usa CUBIL" % path)
        return list(DEFAULT_KEY)
    return key


def save_key(path, key):
    with open(path, "w") as f:
        f.write("".join(key) + "\n")


# ---------------------------------------------------------------- por frame
def classify_mask(mask, clf):
    """Mascara 64x64 -> (etiqueta, distancia).
    None = no hay mano | "?" = mano no reconocida | "C","U","B","I","L" = signo."""
    if mask is None:
        return None, 0.0
    feats = extract(mask)
    if feats is None:
        return None, 0.0
    hu, geo, hog = feats
    return clf.predict_one(hu, geo, hog)


def frame_label(roi_img, bg, lo, hi, clf):
    """ROI -> (etiqueta, distancia, pasos). Hay mano si segment() encuentra un
    contorno de al menos MIN_AREA px."""
    pasos, info = seg.segment(roi_img, bg, lo, hi, seg.DIFF_THRESH)
    mask = pasos["mascara64"] if info is not None else None
    label, dist = classify_mask(mask, clf)
    return label, dist, pasos


# ---------------------------------------------------------------- parametros en vivo
def adjust_params(key, clf):
    """Cambia un parametro del decodificador (o tau) segun la tecla. Devuelve el aviso, o None."""
    if key == ord("1"):
        decoder.WINDOW = max(3, decoder.WINDOW - 1)
    elif key == ord("2"):
        decoder.WINDOW = min(60, decoder.WINDOW + 1)
    elif key == ord("3"):
        decoder.RATIO = round(max(0.5, decoder.RATIO - 0.05), 2)
    elif key == ord("4"):
        decoder.RATIO = round(min(1.0, decoder.RATIO + 0.05), 2)
    elif key == ord("5"):
        decoder.NEUTRAL_FRAMES = max(1, decoder.NEUTRAL_FRAMES - 1)
    elif key == ord("6"):
        decoder.NEUTRAL_FRAMES = min(60, decoder.NEUTRAL_FRAMES + 1)
    elif key == ord("7"):
        decoder.TIMEOUT = max(1.0, decoder.TIMEOUT - 0.5)
    elif key == ord("8"):
        decoder.TIMEOUT = min(30.0, decoder.TIMEOUT + 0.5)
    elif key == ord("9"):
        clf.tau = round(max(0.1, clf.tau - 0.02), 3)
    elif key == ord("0"):
        clf.tau = round(min(3.0, clf.tau + 0.02), 3)
    else:
        return None
    return params_text(clf)


def params_text(clf):
    return "N=%d  ratio=%.2f  M=%d  timeout=%.1fs  tau=%.3f" % (
        decoder.WINDOW, decoder.RATIO, decoder.NEUTRAL_FRAMES, decoder.TIMEOUT, clf.tau)


# ---------------------------------------------------------------- sesion
class Session:
    """Junta el decodificador, el modo administrador y las teclas.
    No usa la camara: se puede probar con etiquetas (demo_offline.py)."""

    def __init__(self, clf, key_file=None):
        self.clf = clf
        self.key_file = key_file if key_file else KEY_FILE
        decoder.KEY = load_key(self.key_file)
        self.dec = SequenceDecoder()
        self.enroll = None         # decodificador temporal del modo administrador
        self.new_key = None        # clave nueva completa, pendiente de ENTER / ESC
        self.note = ""
        self.note_time = -100.0

    def say(self, text, now):
        self.note = text
        self.note_time = now

    def process(self, label, now):
        """Procesa un frame. Devuelve el evento del decodificador (o None)."""
        if self.enroll is None:
            return self.dec.update(label, now)
        if self.new_key is None:
            self.enroll.update(label, now)
            if len(self.enroll.entered) == KEY_LEN:      # copiar ya: el temporal se reinicia solo
                self.new_key = list(self.enroll.entered)
        return None

    def handle_key(self, key, now):
        """Procesa una tecla (codigo de waitKey). Devuelve True si hay que salir."""
        if key == ord("q"):
            return True
        if key == ord("r"):
            if self.enroll is not None:
                self.enroll = SequenceDecoder()
                self.new_key = None
                self.say("Admin: secuencia reiniciada", now)
            elif self.dec.key_state == "LOCKED":
                self.say("Bloqueado: espera a que termine", now)
            else:
                self.dec.key_state = "INPUT"
                self.dec.message = ""
                self.dec.reset_sequence(now)
        elif key == ord("n"):
            if self.enroll is not None:
                pass
            elif self.dec.key_state == "GRANTED":
                self.enroll = SequenceDecoder()
                self.new_key = None
                self.say("Modo admin: haz los 5 signos de la clave nueva", now)
            else:
                self.say("n solo funciona con ACCESO CONCEDIDO", now)
        elif key == 13 and self.new_key is not None:           # ENTER
            decoder.KEY = self.new_key
            save_key(self.key_file, self.new_key)
            self.enroll = None
            self.new_key = None
            self.dec = SequenceDecoder()                        # vuelve a pedir la clave
            self.say("Clave guardada", now)
        elif key == 27 and self.enroll is not None:            # ESC
            self.enroll = None
            self.new_key = None
            self.say("Admin cancelado", now)
        else:
            adjust_params(key, self.clf)                        # la linea de parametros se actualiza sola
        return False


def draw_extra(view, sess, now, roi, dist, hint):
    """Textos que no dibuja decoder.draw_overlay: parametros, avisos, distancia y modo admin."""
    x, y, w, h = roi
    H = view.shape[0]
    if dist > 0:
        cv2.putText(view, "d = %.3f" % dist, (x, y + h + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    if hint:
        cv2.putText(view, hint, (10, H - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    if now - sess.note_time < NOTE_TIME:
        cv2.putText(view, sess.note, (10, H - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(view, params_text(sess.clf), (10, H - 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    if sess.enroll is not None:
        cv2.putText(view, "MODO ADMIN: clave nueva", (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2)
        if sess.new_key is not None:
            cv2.putText(view, " - ".join(sess.new_key), (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(view, "ENTER guardar | ESC cancelar | r repetir", (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_view(frame, sess, now, fps, roi, label, dist, hint):
    """Dibuja todo el feedback sobre una copia del frame."""
    view = frame.copy()
    if sess.enroll is not None and sess.new_key is not None:
        # clave nueva completa: no se usa el decodificador temporal (mostraria GRANTED y la clave vieja)
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(view, "ADMIN", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        for i in range(KEY_LEN):
            cv2.circle(view, (20 + 40 * i, 60), 12, (255, 255, 255), -1)
        cv2.putText(view, "FPS: %.1f" % fps, (view.shape[1] - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        shown = sess.enroll if sess.enroll is not None else sess.dec
        draw_overlay(view, shown, now, fps, roi, label)
    draw_extra(view, sess, now, roi, dist, hint)
    return view


# ---------------------------------------------------------------- programa principal
def main():
    clf = Classifier.load(MODEL_FILE)
    sess = Session(clf)
    size, map1, map2 = load_maps()
    cap = open_camera()
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return
    print("Driver:", driver_state(cap))
    print("Clave actual: %s | %s" % ("-".join(decoder.KEY), params_text(clf)))

    roi = list(seg.ROI)
    bg = None
    bg_chroma = None
    skin_center = None
    skin_std = None
    collecting = None
    bg_frames = []
    skin_frames = []
    times = []                 # para los FPS en pantalla
    n_ready = 0                # frames con el sistema calibrado (para los FPS medios)
    t_ready = None
    proc_time = 0.0            # tiempo de procesado (segmentacion + clasificador + decoder)

    cv2.namedWindow("demo")
    cv2.moveWindow("demo", 0, 0)
    cv2.namedWindow("mascara")
    cv2.moveWindow("mascara", 660, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame")
            break
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        now = time.time()
        times.append(now)
        if len(times) > N_FPS:
            times.pop(0)
        fps = 0.0
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])

        roi_img = seg.get_roi(frame, roi)

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

        # reconocimiento
        ready = bg is not None and skin_center is not None and collecting is None
        label, dist, pasos = None, 0.0, None
        wall_in_skin = False
        t0 = time.time()
        if ready:
            lo, hi = seg.skin_range(skin_center, skin_std, seg.MIN_HALF)
            wall_in_skin = bool(np.all((bg_chroma >= lo) & (bg_chroma <= hi)))
            label, dist, pasos = frame_label(roi_img, bg, lo, hi, clf)
        event = sess.process(label, now)
        if ready:
            proc_time += time.time() - t0
            n_ready += 1
            if t_ready is None:
                t_ready = now
        if event is not None:
            print("[%s] evento: %s" % (time.strftime("%H:%M:%S"), event))

        # pantalla
        if collecting == "bg":
            hint = "Capturando fondo (%d/%d): no muevas nada" % (len(bg_frames), seg.N_BG)
        elif collecting == "skin":
            hint = "Capturando piel (%d/%d): palma quieta" % (len(skin_frames), seg.N_SKIN)
        elif bg is None:
            hint = "PASO 1: quita la mano de la ROI y pulsa b"
        elif skin_center is None:
            hint = "PASO 2: palma sobre el cuadrado cian y pulsa p"
        elif wall_in_skin:
            hint = "AVISO: el fondo cae en el rango de piel (mas luz)"
        else:
            hint = ""
        view = draw_view(frame, sess, now, fps, roi, label, dist, hint)
        if not ready:
            x, y, w, h = roi
            sx, sy, sw, sh = seg.sample_box(w, h)
            cv2.rectangle(view, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 255, 0), 1)
        cv2.imshow("demo", view)
        small = np.zeros((seg.OUT, seg.OUT), np.uint8) if pasos is None else pasos["mascara64"]
        cv2.imshow("mascara", cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST))

        # teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord("b") and sess.enroll is None:
            bg_frames = []
            collecting = "bg"
        elif key == ord("p") and sess.enroll is None:
            skin_frames = []
            collecting = "skin"
        elif key != 255:
            if sess.handle_key(key, now):
                break

    cap.release()
    cv2.destroyAllWindows()
    if n_ready > 1:
        elapsed = time.time() - t_ready
        print("FPS medios (sistema calibrado): %.1f | tiempo medio de procesado: %.1f ms/frame"
              % (n_ready / elapsed, 1000 * proc_time / n_ready))
    print("Clave final: %s | %s" % ("-".join(decoder.KEY), params_text(clf)))


if __name__ == "__main__":
    main()
