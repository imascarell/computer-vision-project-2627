# decoder.py - Decodificador de la secuencia de signos (sistema de seguridad)
#
# Recibe, frame a frame, la etiqueta que da el clasificador:
#   "C", "U", "B", "I", "L"  -> signo reconocido
#   "?"                      -> hay mano pero no es ninguno de los 5 signos
#   None                     -> no hay mano en la ROI
#
# Maquina de estados en dos niveles:
#   Estado de la clave: "INPUT" -> "GRANTED" / "DENIED" / "LOCKED"
#   Estado del signo:   "WAIT_HAND" -> "DETECTING" -> "WAIT_NEUTRAL" -> "WAIT_HAND"

import cv2

# ------------------------------------------------------------------ parametros
KEY = ["C", "U", "B", "I", "L"]   # clave correcta
WINDOW = 15            # frames que se miran para confirmar un signo (~0.5 s a 30 FPS)
RATIO = 0.8            # el signo debe aparecer en al menos el 80 % de esos frames
NEUTRAL_FRAMES = 10    # frames sin mano necesarios antes del siguiente signo
TIMEOUT = 5.0          # segundos sin confirmar signo -> se borra la secuencia
DENIED_TIME = 2.0      # segundos que se muestra "ACCESO DENEGADO"
MAX_FAILS = 3          # fallos seguidos antes de bloquear
LOCK_TIME = 30.0       # segundos de bloqueo


class SequenceDecoder:

    def __init__(self):
        self.key_state = "INPUT"
        self.fails = 0
        self.state_time = 0.0      # instante en que se entro en el estado actual
        self.message = ""
        self.reset_sequence(0.0)

    def reset_sequence(self, now):
        """Olvida lo introducido (tecla 'r' o timeout). No cambia la clave."""
        self.entered = []          # signos confirmados hasta ahora (S0..S5)
        self.error = False         # True si algun signo no coincide con la clave
        self.last_sign_time = now
        self.sign_state = "WAIT_HAND"
        self.window = []           # ultimas etiquetas mientras se detecta un signo
        self.no_hand = 0           # frames seguidos sin mano

    # -------------------------------------------------------------- por frame
    def update(self, label, now):
        """Procesa un frame. Devuelve el signo confirmado, "GRANTED", "DENIED",
        "LOCKED", "TIMEOUT" o None si no pasa nada."""

        # 1) Estados de la clave que no leen signos
        if self.key_state == "GRANTED":
            return None

        if self.key_state == "DENIED":
            if now - self.state_time > DENIED_TIME:
                self.key_state = "INPUT"
                self.message = ""
                self.reset_sequence(now)
            return None

        if self.key_state == "LOCKED":
            if now - self.state_time > LOCK_TIME:
                self.key_state = "INPUT"
                self.fails = 0
                self.message = ""
                self.reset_sequence(now)
            return None

        # 2) Timeout entre signos
        if len(self.entered) > 0 and now - self.last_sign_time > TIMEOUT:
            self.reset_sequence(now)
            self.message = "Timeout: secuencia borrada"
            return "TIMEOUT"

        # 3) Nivel signo: intentar confirmar un signo
        sign = self.update_sign(label)
        if sign is None:
            return None

        # 4) Nivel clave: se ha confirmado un signo
        position = len(self.entered)          # posicion que ocupa este signo (0..4)
        self.entered.append(sign)
        self.last_sign_time = now
        if sign != KEY[position]:
            self.error = True                 # no se avisa todavia: se decide al final

        if len(self.entered) < len(KEY):
            return sign

        # Se han introducido los 5 signos -> verificar
        if not self.error:
            self.key_state = "GRANTED"
            self.fails = 0
            self.message = "ACCESO CONCEDIDO"
            return "GRANTED"

        self.fails = self.fails + 1
        self.state_time = now
        if self.fails >= MAX_FAILS:
            self.key_state = "LOCKED"
            self.message = "BLOQUEADO"
            return "LOCKED"
        self.key_state = "DENIED"
        self.message = "ACCESO DENEGADO"
        return "DENIED"

    def update_sign(self, label):
        """Nivel signo. Devuelve la letra cuando se confirma, si no None."""
        hand = label is not None

        if self.sign_state == "WAIT_NEUTRAL":
            # Esperar a que se retire la mano para no contar dos veces el mismo signo
            if hand:
                self.no_hand = 0
            else:
                self.no_hand = self.no_hand + 1
            if self.no_hand >= NEUTRAL_FRAMES:
                self.sign_state = "WAIT_HAND"
            return None

        if self.sign_state == "WAIT_HAND":
            if not hand:
                return None
            self.sign_state = "DETECTING"
            self.window = []
            self.no_hand = 0

        # DETECTING
        if not hand:
            self.no_hand = self.no_hand + 1
            if self.no_hand >= NEUTRAL_FRAMES:      # la mano se fue sin confirmar
                self.sign_state = "WAIT_HAND"
                self.window = []
            return None

        self.no_hand = 0
        self.window.append(label)
        if len(self.window) > WINDOW:
            self.window.pop(0)                      # quedarse con los ultimos WINDOW
        if len(self.window) < WINDOW:
            return None

        best, count = self.most_common()
        if best != "?" and count >= RATIO * WINDOW:
            self.sign_state = "WAIT_NEUTRAL"
            self.window = []
            self.no_hand = 0
            return best
        return None

    def most_common(self):
        """Etiqueta mas repetida en la ventana y cuantas veces aparece."""
        best, count = None, 0
        for lab in set(self.window):
            n = self.window.count(lab)
            if n > count:
                best, count = lab, n
        return best, count


# ---------------------------------------------------------------- dibujo
def draw_overlay(frame, dec, now, fps, roi, label):
    """Dibuja el feedback en el frame. roi = (x, y, w, h)."""
    x, y, w, h = roi

    # Recuadro de la ROI: gris sin mano, amarillo detectando, verde confirmado, rojo desconocido
    if label == "?":
        color = (0, 0, 255)
    elif dec.sign_state == "DETECTING":
        color = (0, 220, 255)
    elif dec.sign_state == "WAIT_NEUTRAL":
        color = (0, 200, 0)
    else:
        color = (160, 160, 160)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label is not None:
        cv2.putText(frame, "Signo: " + label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Barra de confirmacion del signo
    if dec.sign_state == "DETECTING" and len(dec.window) > 0:
        best, count = dec.most_common()
        if best != "?":
            p = min(1.0, count / (RATIO * WINDOW))
            cv2.rectangle(frame, (x, y + h + 6), (x + int(w * p), y + h + 14), (0, 220, 255), -1)

    # Progreso de la clave: circulos llenos = signos introducidos (sin decir si son correctos)
    for i in range(len(KEY)):
        if i < len(dec.entered):
            cv2.circle(frame, (20 + 40 * i, 60), 12, (255, 255, 255), -1)
        else:
            cv2.circle(frame, (20 + 40 * i, 60), 12, (255, 255, 255), 2)

    # Tiempo restante (timeout o bloqueo)
    if dec.key_state == "LOCKED":
        left = LOCK_TIME - (now - dec.state_time)
        cv2.putText(frame, "%.0f s" % left, (230, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif len(dec.entered) > 0 and dec.key_state == "INPUT":
        left = TIMEOUT - (now - dec.last_sign_time)
        cv2.putText(frame, "%.1f s" % left, (230, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Traduccion de la clave al conceder acceso
    if dec.key_state == "GRANTED":
        cv2.putText(frame, " - ".join(KEY), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Estado y mensaje
    if dec.key_state == "GRANTED":
        msg_color = (0, 255, 0)
    elif dec.key_state in ("DENIED", "LOCKED"):
        msg_color = (0, 0, 255)
    else:
        msg_color = (255, 255, 255)
    cv2.putText(frame, dec.key_state + "  S" + str(len(dec.entered)), (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2)
    cv2.putText(frame, dec.message, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)

    # FPS (lo pide el enunciado)
    cv2.putText(frame, "FPS: %.1f" % fps, (frame.shape[1] - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
