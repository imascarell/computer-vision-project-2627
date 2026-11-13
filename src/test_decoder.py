# test_decoder.py - Prueba el decodificador con etiquetas simuladas (sin camara)
from decoder import SequenceDecoder

FPS = 30
DT = 1 / FPS


def feed(dec, letters, t, hold=20, gap=12, noise=False):
    """Simula hacer cada signo 'hold' frames y retirar la mano 'gap' frames."""
    events = []
    for s in letters:
        for k in range(hold):
            label = s
            if noise and k % 7 == 3:
                label = "?"            # algun frame mal clasificado
            e = dec.update(label, t)
            t = t + DT
            if e:
                events.append(e)
        for k in range(gap):
            e = dec.update(None, t)
            t = t + DT
            if e:
                events.append(e)
    return events, t


t = 0.0

d = SequenceDecoder()
ev, t = feed(d, "CUBIL", t, noise=True)
print("Clave correcta con ruido:", ev)
assert d.key_state == "GRANTED"

d = SequenceDecoder()
ev, t = feed(d, "CUBLI", t)
print("Orden incorrecto:", ev)
assert ev[-1] == "DENIED"

d = SequenceDecoder()
for i in range(3):
    ev, t = feed(d, "LIBUC", t)
    t = t + 2.1
    d.update(None, t)
print("3 fallos ->", d.key_state)
assert d.key_state == "LOCKED"
t = t + 31
d.update(None, t)
print("Tras 30 s ->", d.key_state)
assert d.key_state == "INPUT"

d = SequenceDecoder()
ev, t = feed(d, "CU", t)
t = t + 6
print("Timeout:", d.update(None, t), "| signos:", len(d.entered))
assert len(d.entered) == 0

d = SequenceDecoder()
ev, t = feed(d, "C", t, hold=100, gap=0)
print("C mantenida mucho tiempo:", ev)
assert len(d.entered) == 1

d = SequenceDecoder()
ev, t = feed(d, "???", t)
print("Signos desconocidos:", ev)
assert len(d.entered) == 0

d = SequenceDecoder()
ev, t = feed(d, "CUB", t)
d.reset_sequence(t)
print("Reset -> signos:", len(d.entered))
assert len(d.entered) == 0

print("TODO OK")
