# entrenar.py - Entrena el clasificador final con TODO el dataset y lo guarda (Fase 4)
#
# Configuracion elegida en eval_clasificador.py (evaluacion anidada): HOG, k=3, tau = P95.
# Guarda data/modelo.npz; la demo (Fase 5) lo carga con Classifier.load().
# El modelo guarda las distancias intra-clase, asi que tau se puede cambiar sin reentrenar.
#
# Uso:  python entrenar.py

import os
import numpy as np
from classifier import Classifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "data", "dataset.npz")
MODELO = os.path.join(HERE, "data", "modelo.npz")


def main():
    d = np.load(DATASET)
    c = Classifier(feats="hog", k=3, pct=95)
    c.fit(d["hu"], d["geo"], d["hog"], d["labels"], d["bloque"])
    c.save(MODELO)

    print("Muestras: %d | configuracion: %s k=%d | s_hog=%.3f" % (len(c.y), c.feats, c.k, c.s_hog))
    print("Distancias intra-clase (otros bloques): min %.3f | mediana %.3f | max %.3f"
          % (c.intra.min(), np.median(c.intra), c.intra.max()))
    for p in (90, 95, 99):
        print("   tau P%d = %.3f%s" % (p, np.percentile(c.intra, p), "   <-- usado" if p == c.pct else ""))

    # comprobacion: el modelo cargado da lo mismo que el original
    c2 = Classifier.load(MODELO)
    p1, d1 = c.predict(d["hu"], d["geo"], d["hog"])
    p2, d2 = c2.predict(d["hu"], d["geo"], d["hog"])
    print("Guardado en %s" % MODELO)
    print("Modelo cargado igual que el original: %s" % ((p1 == p2).all() and np.allclose(d1, d2)))


if __name__ == "__main__":
    main()
