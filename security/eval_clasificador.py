# eval_clasificador.py - Evaluacion del clasificador k-NN (Fase 4)
#
# Evaluacion ANIDADA con bloques (cada bloque = una condicion: centrada, mas/menos antebrazo,
# giro izquierda, giro derecha), para no inflar el resultado:
#   - Bucle externo: se deja un bloque entero fuera como test (4 vueltas). Cada muestra se
#     predice una sola vez con un modelo que NO ha visto su bloque.
#   - Bucle interno: dentro de los 3 bloques de entrenamiento se elige la configuracion
#     (features, k, w) dejando de nuevo un bloque fuera. El test externo no interviene.
#   - Si varias configuraciones quedan a menos de 1 punto de la mejor, se elige la mas simple.
#   - z-score, escalas y tau se calculan siempre solo con el entrenamiento de cada vuelta.
#
# Uso:  python eval_clasificador.py             (solo imprime, no escribe nada)
#       python eval_clasificador.py --guardar   (ademas guarda results/matriz_confusion_fase4.csv)

import os
import sys
from collections import Counter
import numpy as np
from classifier import Classifier

HERE = os.path.dirname(os.path.abspath(__file__))
LETRAS = ["C", "U", "B", "I", "L"]
PCTS = [90, 95, 99]                 # percentiles de tau que se comparan
MARGEN = 0.01                       # empate: a menos de 1 punto de la mejor
# de mas simple a mas compleja (a igualdad se elige la primera)
CONFIGS = [("hog", 3, 1.0), ("hog", 5, 1.0), ("geohu", 3, 1.0), ("geohu", 5, 1.0)] + \
          [("comb", k, w) for w in (0.5, 1.0, 2.0) for k in (3, 5)]


def nombre(cfg):
    feats, k, w = cfg
    return "%s k=%d" % (feats, k) + (" w=%.1f" % w if feats == "comb" else "")


def ajustar(d, idx, cfg):
    """Entrena con las muestras idx usando la configuracion cfg."""
    feats, k, w = cfg
    return Classifier(feats, k, w).fit(d["hu"][idx], d["geo"][idx], d["hog"][idx],
                                       d["labels"][idx], d["bloque"][idx])


def predecir(c, d, idx, reject=False):
    return c.predict(d["hu"][idx], d["geo"][idx], d["hog"][idx], reject)


def bloque_fuera(d, idx, cfg):
    """Dejando un bloque fuera cada vez (solo entre las muestras idx): predicciones de idx."""
    pred = np.empty(len(idx), dtype="<U1")
    for b in np.unique(d["bloque"][idx]):
        test = d["bloque"][idx] == b
        c = ajustar(d, idx[~test], cfg)
        pred[test] = predecir(c, d, idx[test])[0]
    return pred


def elegir(d, idx):
    """Bucle interno: configuracion con mejor precision (empates -> la mas simple)."""
    accs = [np.mean(bloque_fuera(d, idx, cfg) == d["labels"][idx]) for cfg in CONFIGS]
    mejor = max(accs)
    for cfg, a in zip(CONFIGS, accs):
        if a >= mejor - MARGEN:
            return cfg, a


def main():
    d = np.load(os.path.join(HERE, "data", "dataset.npz"))
    y, bloque = d["labels"], d["bloque"]
    todos = np.arange(len(y))

    # ---------------------------------------------------------- evaluacion anidada
    pred = np.empty(len(y), dtype="<U1")
    dist = np.zeros(len(y))
    tau = {p: np.zeros(len(y)) for p in PCTS}       # tau de la vuelta que predijo cada muestra
    elegidas = []
    print("== 1. Configuracion elegida en cada vuelta (bucle interno) ==")
    for t in np.unique(bloque):
        train = todos[bloque != t]
        test = todos[bloque == t]
        cfg, acc_in = elegir(d, train)
        c = ajustar(d, train, cfg)
        pred[test], dist[test] = predecir(c, d, test)
        for p in PCTS:
            tau[p][test] = np.percentile(c.intra, p)
        elegidas.append(cfg)
        print("   bloque %d fuera: %-18s (precision interna %.1f %%)" % (t, nombre(cfg), 100 * acc_in))

    ok = pred == y
    print("\n== 2. Resultado externo SIN rechazo (cada muestra con un modelo que no vio su bloque) ==")
    print("   Global: %.1f %% (%d/%d)" % (100 * ok.mean(), ok.sum(), len(y)))
    print("   Por bloque: " + "  ".join("%d: %.0f %%" % (b, 100 * ok[bloque == b].mean())
                                        for b in np.unique(bloque)))

    cm = np.zeros((5, 5), int)                      # filas = letra real, columnas = predicha
    for real, pr in zip(y, pred):
        cm[LETRAS.index(real), LETRAS.index(pr)] += 1
    print("\n== 3. Matriz de confusion (filas = real, columnas = predicha) ==")
    print("        " + " ".join("%4s" % l for l in LETRAS) + "   recall  precision")
    for i, l in enumerate(LETRAS):
        recall = cm[i, i] / cm[i].sum()
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        marca = "" if recall > 0.90 else "  <-- NO supera el 90 %"
        print("   %s    " % l + " ".join("%4d" % v for v in cm[i]) +
              "   %5.1f %%  %5.1f %%%s" % (100 * recall, 100 * prec, marca))
    u, b = LETRAS.index("U"), LETRAS.index("B")
    print("   Confusiones U<->B: U tomada por B = %d, B tomada por U = %d" % (cm[u, b], cm[b, u]))

    print("\n== 4. Rechazo con distintos tau (sobre letras VALIDAS; menos rechazadas = mejor) ==")
    print("   percentil | rechazadas | aceptadas y bien | aceptadas y mal")
    for p in PCTS:
        rech = dist > tau[p]
        bien = ~rech & ok
        mal = ~rech & ~ok
        print("   P%-8d | %9.1f %% | %15.1f %% | %13.1f %%" %
              (p, 100 * rech.mean(), 100 * bien.mean(), 100 * mal.mean()))

    # ---------------------------------------------------------- descriptiva
    print("\n== 5. (Solo descriptivo, NO se usa para elegir) bloque-fuera de cada configuracion ==")
    for cfg in CONFIGS:
        a = np.mean(bloque_fuera(d, todos, cfg) == y)
        print("   %-18s %.1f %%" % (nombre(cfg), 100 * a))

    # ---------------------------------------------------------- falsa aceptacion (aproximada)
    modal = Counter(elegidas).most_common()
    mejor_n = modal[0][1]
    cfg_final = [c for c in CONFIGS if c in dict(modal) and dict(modal)[c] == mejor_n][0]
    print("\n== 6. Falsa aceptacion aproximada: una letra se deja fuera del entrenamiento ==")
    print("   (configuracion mas elegida: %s; se mide cuantos frames de la letra ausente NO se rechazan)"
          % nombre(cfg_final))
    print("   letra fuera | " + " | ".join("P%d" % p for p in PCTS))
    for letra in LETRAS:
        train = todos[y != letra]
        test = todos[y == letra]
        c = ajustar(d, train, cfg_final)
        _, dd = predecir(c, d, test)
        fila = ["%5.1f %%" % (100 * np.mean(dd <= np.percentile(c.intra, p))) for p in PCTS]
        print("   %-11s | " % letra + " | ".join(fila))
    print("   (es una aproximacion: las otras letras se parecen a la clave mas que un gesto cualquiera)")

    if "--guardar" in sys.argv:
        carpeta = os.path.join(HERE, "results")
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, "matriz_confusion_fase4.csv")
        np.savetxt(ruta, cm, fmt="%d", delimiter=",", header=",".join(LETRAS), comments="")
        print("\nGuardado: " + ruta)


if __name__ == "__main__":
    main()
