# demo_offline.py - Prueba de la Fase 5 SIN camara
#
# Alimenta la misma logica que demo.py (clasificador + decoder + teclas)
# con mascaras del dataset, y el bucle completo de demo.py con una camara simulada.
# Los datos son los de entrenamiento: esto valida la LOGICA, no la precision (eso es la Fase 6).
#
# Uso:  python demo_offline.py

import os
import tempfile
import time
import cv2
import numpy as np

import decoder
import demo
import segmentation as seg
from classifier import Classifier
from features import DEFAULT_CAPTURES

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
DT = 1 / FPS
DEFAULTS = (decoder.WINDOW, decoder.RATIO, decoder.NEUTRAL_FRAMES, decoder.TIMEOUT)

rng = np.random.RandomState(0)
data = np.load(os.path.join(HERE, "data", "dataset.npz"))
MASKS = {l: data["masks"][data["labels"] == l] for l in demo.LETTERS}
CIRCLE = np.zeros((64, 64), np.uint8)
cv2.circle(CIRCLE, (32, 32), 22, 255, -1)          # forma que el clasificador rechaza ("?")
clf = Classifier.load(demo.MODEL_FILE)


def reset_params():
    decoder.WINDOW, decoder.RATIO, decoder.NEUTRAL_FRAMES, decoder.TIMEOUT = DEFAULTS
    clf.tau = float(np.load(demo.MODEL_FILE)["tau"])


def new_session():
    reset_params()
    return demo.Session(clf)


def feed(sess, letters, t, hold=20, gap=12, noise=False, circle=False):
    """Simula hacer cada signo 'hold' frames (mascaras reales del dataset) y retirar la mano."""
    events = []
    for s in letters:
        for k in range(hold):
            if circle:
                mask = CIRCLE
            elif noise and k % 7 == 3:
                mask = CIRCLE                           # un frame mal clasificado
            else:
                mask = MASKS[s][rng.randint(len(MASKS[s]))]
            label, dist = demo.classify_mask(mask, clf)
            e = sess.process(label, t)
            t += DT
            if e:
                events.append(e)
        for k in range(gap):
            label, dist = demo.classify_mask(None, clf)
            e = sess.process(label, t)
            t += DT
            if e:
                events.append(e)
    return events, t


t = 0.0

# 1) clave correcta con ruido
s = new_session()
ev, t = feed(s, "CUBIL", t, noise=True)
print("1) CUBIL con ruido:", ev)
assert s.dec.key_state == "GRANTED"

# 2) secuencias incorrectas: todas se deniegan
bad = ["CUBLI", "LIBUC", "CUBIB", "UCBIL", "CCCCC", "BBBBB", "CUBIU"]
for _ in range(10):
    q = "".join(rng.choice(list("CUBIL"), 5))
    if q != "CUBIL":
        bad.append(q)
for q in bad:
    s = new_session()
    ev, t = feed(s, q, t)
    assert ev[-1] == "DENIED", (q, ev)
print("2) %d secuencias incorrectas: todas DENIED" % len(bad))

# 3) tres fallos seguidos -> LOCKED, r no desbloquea, a los 30 s vuelve
s = new_session()
for i in range(3):
    ev, t = feed(s, "LIBUC", t)
    t += decoder.DENIED_TIME + 0.1
    s.process(None, t)
assert s.dec.key_state == "LOCKED"
s.handle_key(ord("r"), t)
assert s.dec.key_state == "LOCKED"
t += decoder.LOCK_TIME + 1
s.process(None, t)
assert s.dec.key_state == "INPUT"
print("3) 3 fallos -> LOCKED; r no lo salta; tras 30 s -> INPUT")

# 4) mano no reconocida ("?") nunca confirma signos
s = new_session()
ev, t = feed(s, "???", t, circle=True)
assert len(s.dec.entered) == 0 and ev == []
print("4) forma no reconocida: 0 signos confirmados")

# 5) r tras GRANTED vuelve a pedir la clave
s = new_session()
ev, t = feed(s, "CUBIL", t)
assert s.dec.key_state == "GRANTED"
s.handle_key(ord("r"), t)
assert s.dec.key_state == "INPUT" and len(s.dec.entered) == 0
ev, t = feed(s, "CUBIL", t)
assert s.dec.key_state == "GRANTED"
print("5) r re-bloquea tras GRANTED y CUBIL vuelve a abrir")

# 6) timeout
s = new_session()
ev, t = feed(s, "CU", t)
t += decoder.TIMEOUT + 0.5
assert s.process(None, t) == "TIMEOUT" and len(s.dec.entered) == 0
print("6) timeout borra la secuencia")

# 7) CUBIL es la unica clave: ninguna tecla la cambia ni acepta otra
s = new_session()
for k in "nN\r\x1b":
    s.handle_key(ord(k), t)
assert decoder.KEY == list("CUBIL")
ev, t = feed(s, "LLUBC", t)
assert ev[-1] == "DENIED"
s = new_session()
ev, t = feed(s, "CUBIL", t)
assert ev[-1] == "GRANTED"
s.handle_key(ord("n"), t)                                # n no hace nada
assert s.dec.key_state == "GRANTED" and decoder.KEY == list("CUBIL")
print("7) la tecla n no cambia la clave; solo CUBIL concede")

# 8) ajuste de parametros en vivo (con limites)
s = new_session()
tau0 = float(np.load(demo.MODEL_FILE)["tau"])
for k in "2" * 3 + "4" * 2 + "6" + "8" + "0":
    s.handle_key(ord(k), t)
assert decoder.WINDOW == 18 and decoder.RATIO == 0.9 and decoder.NEUTRAL_FRAMES == 11
assert decoder.TIMEOUT == 5.5 and clf.tau == round(tau0 + 0.02, 3)
for k in "1" * 100 + "3" * 100 + "5" * 100 + "7" * 100 + "9" * 100:
    s.handle_key(ord(k), t)
assert (decoder.WINDOW, decoder.RATIO, decoder.NEUTRAL_FRAMES, decoder.TIMEOUT, clf.tau) == (3, 0.5, 1, 1.0, 0.1)
print("8) teclas 1-0 ajustan N, ratio, M, timeout y tau, con limites")
reset_params()

# 9) velocidad de la parte sin camara (clasificador + decoder + overlay)
s = new_session()
frame = np.zeros((480, 640, 3), np.uint8)
n = 600
t0 = time.time()
for i in range(n):
    m = MASKS["CUBIL"[(i // 40) % 5]][i % 40]
    label, dist = demo.classify_mask(m if (i % 40) < 30 else None, clf)
    s.process(label, i * DT)
    demo.draw_view(frame, s, i * DT, 30.0, seg.ROI, label, dist, "")
dt = (time.time() - t0) / n
print("9) clasificador + decoder + overlay: %.2f ms/frame" % (dt * 1000))

# 10) pipeline real (segment -> features -> clasificador) sobre las capturas de la Fase 2
caps = []
for letter, num in DEFAULT_CAPTURES:
    img = cv2.imread(os.path.join(HERE, "captures", "seg_%02d_01_roi.png" % num))
    caps.append((letter, img))
lo, hi = data["skin_lo"][0], data["skin_hi"][0]
t_seg = []
res = []
for letter, img in caps:
    wall = np.median(img[:20, :20].reshape(-1, 3), axis=0).astype(np.uint8)
    bg = np.full_like(img, wall)                         # fondo liso aproximado (no hay ROI vacia guardada)
    t0 = time.time()
    label, dist, pasos = demo.frame_label(img, bg, lo, hi, clf)
    t_seg.append(time.time() - t0)
    res.append((letter, label, dist))
print("10) capturas reales de la Fase 2 -> etiqueta (esperada, obtenida, distancia):")
for letter, label, dist in res:
    print("      %s -> %s  d=%.3f" % (letter, label, dist))
print("    segment+features+clasificador: %.1f ms/frame (este entorno, orientativo)" % (1000 * np.mean(t_seg)))

# 11) bucle completo de demo.main() con camara simulada
reset_params()
_tmp = tempfile.mkdtemp()
demo.LOG_FILE = os.path.join(_tmp, "intentos.csv")
demo.FPS_FILE = os.path.join(_tmp, "fps.csv")
rois = {l: cv2.imread(os.path.join(HERE, "captures", "seg_%02d_01_roi.png" % n)) for l, n in DEFAULT_CAPTURES}
hands = {l: cv2.imread(os.path.join(HERE, "captures", "seg_%02d_05_morfologia.png" % n), 0) > 0
         for l, n in DEFAULT_CAPTURES}
wall = np.array([170, 110, 70], np.uint8)            # pared azulada: fuera del rango de piel
X, Y, W, H = seg.ROI


PALM = np.median(rois["B"][hands["B"]], axis=0).astype(np.uint8)    # color tipico de la mano


def palm_canvas():
    """Palma de la B y, en el cuadrado de calibracion, color de piel uniforme."""
    f = canvas("B")
    sx, sy, sw, sh = seg.sample_box(W, H)
    f[Y + sy:Y + sy + sh, X + sx:X + sx + sw] = PALM
    return f


def canvas(letter=None):
    """Pared lisa; con letra, se pegan encima solo los pixeles de la mano de esa captura."""
    f = np.full((480, 640, 3), wall, np.uint8)
    if letter is not None:
        roi = f[Y:Y + H, X:X + W]
        roi[hands[letter]] = rois[letter][hands[letter]]
    return f


class FakeCap:
    def __init__(self, schedule):
        self.frames = [f for f, k in schedule]
        self.keys = [k for f, k in schedule]
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        self.i += 1
        return True, self.frames[self.i - 1]

    def get(self, p):
        return 0.0

    def set(self, p, v):
        return True

    def release(self):
        pass


def schedule_for(letters):
    sch = []
    sch += [(canvas(), None)] * 2 + [(canvas(), ord("b"))] + [(canvas(), None)] * (seg.N_BG + 2)
    sch += [(palm_canvas(), ord("p"))] + [(palm_canvas(), None)] * (seg.N_SKIN + 2)
    sch += [(canvas(), None)] * 15
    sch += [(canvas(), ord("0"))] * 7                  # tau 0.674 -> 0.814 (la C de la Fase 2 da d = 0.80)
    for s_ in letters:
        sch += [(canvas(s_), None)] * 20 + [(canvas(), None)] * 12
    sch += [(canvas(), ord("q"))]
    return sch


def run_fake(letters):
    cap = FakeCap(schedule_for(letters))
    box = []

    class S(demo.Session):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            box.append(self)

    demo.Session = S
    demo.open_camera = lambda: cap
    demo.driver_state = lambda c: "simulada"
    cv2.namedWindow = lambda *a: None
    cv2.moveWindow = lambda *a: None
    cv2.destroyAllWindows = lambda: None
    last = {}
    cv2.imshow = lambda name, img: last.__setitem__(name, img)
    cv2.waitKey = lambda ms: cap.keys[cap.i - 1] if cap.keys[cap.i - 1] else 255
    demo.main()
    return box[0], last


sess, last = run_fake("CUBIL")
print("11) bucle demo.main() con camara simulada, CUBIL:", sess.dec.key_state)
assert sess.dec.key_state == "GRANTED"
cv2.imwrite(os.path.join(tempfile.gettempdir(), "demo_granted.png"), last["demo"])
sess, last = run_fake("CUBLI")
print("    CUBLI:", sess.dec.key_state)
assert sess.dec.key_state == "DENIED"

# 12) registro de intentos y metricas (CSV temporal)
import subprocess
import sys
tmp = os.path.join(tempfile.mkdtemp(), "intentos.csv")
reset_params()
s = demo.Session(clf, tmp)
s.handle_key(ord("c"), t)
ev, t = feed(s, "CUBIL", t)                                   # correcta aceptada
s.handle_key(ord("r"), t)
ev, t = feed(s, "CUBLI", t)                                   # tipo correcta pero falla
t += decoder.DENIED_TIME + 0.1
s.process(None, t)
s.handle_key(ord("x"), t)
ev, t = feed(s, "LIBUC", t)                                   # incorrecta rechazada
t += decoder.DENIED_TIME + 0.1
s.process(None, t)
ev, t = feed(s, "CU", t)                                      # incorrecta: timeout
t += decoder.TIMEOUT + 0.5
s.process(None, t)
rows = open(tmp).read().strip().split("\n")
assert rows[0] == demo.LOG_HEADER and len(rows) == 5, rows
cols = [r.split(",") for r in rows[1:]]
assert [(c[1], c[2], c[4]) for c in cols] == [("correcta", "GRANTED", "CUBIL"),
    ("correcta", "DENIED", "CUBLI"), ("incorrecta", "DENIED", "LIBUC"), ("incorrecta", "TIMEOUT", "CU")], cols
assert abs(float(cols[0][3]) - 142 * DT) < 0.05               # 4 signos de 32 frames + 15 del ultimo = 142 frames
assert s.counts == {"correcta": 2, "incorrecta": 2}
out = subprocess.run([sys.executable, os.path.join(HERE, "metricas.py"), tmp], capture_output=True, text=True).stdout
assert "FRR = 1 / 2 = 50.0 %" in out and "FAR = 0 / 2 = 0.0 %" in out, out
print("12) registro de intentos y metricas OK (4 intentos: GRANTED, DENIED, DENIED, TIMEOUT)")
print(out)
reset_params()

print("TODO OK")
