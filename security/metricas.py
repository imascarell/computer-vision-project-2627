# metricas.py - Resume las pruebas de la Fase 6 (sin camara)
#
# Lee results/intentos.csv (lo escribe demo.py: un intento por fila) y results/fps.csv.
#   FRR = % de intentos con la clave CORRECTA que no dieron GRANTED (falso rechazo)
#   FAR = % de intentos con clave INCORRECTA que dieron GRANTED (falsa aceptacion)
#   Tiempo = segundos desde que aparece la mano hasta GRANTED (solo intentos correctos aceptados)
#
# Uso:  python metricas.py                 (usa results/intentos.csv)
#       python metricas.py otro.csv        (otro fichero de intentos)

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "results", "intentos.csv")
FPS_FILE = os.path.join(HERE, "results", "fps.csv")


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def percent(n, total):
    if total == 0:
        return 0.0
    return 100.0 * n / total


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    rows = read_rows(path)
    if not rows:
        print("No hay intentos en %s" % path)
        return
    good = [r for r in rows if r["tipo"] == "correcta"]
    bad = [r for r in rows if r["tipo"] == "incorrecta"]
    free = len(rows) - len(good) - len(bad)
    print("Intentos: %d con clave correcta | %d con clave incorrecta | %d sin tipo (no cuentan)"
          % (len(good), len(bad), free))

    # FRR
    if good:
        accepted = [r for r in good if r["resultado"] == "GRANTED"]
        rejected = [r for r in good if r["resultado"] != "GRANTED"]
        print("\nFRR = %d / %d = %.1f %%" % (len(rejected), len(good), percent(len(rejected), len(good))))
        for r in rejected:
            print("   fallo: %s (secuencia leida: %s)" % (r["resultado"], r["secuencia"] if r["secuencia"] else "-"))
        if accepted:
            t = [float(r["segundos"]) for r in accepted]
            print("Tiempo medio de introduccion: %.1f s (min %.1f, max %.1f, n = %d)"
                  % (sum(t) / len(t), min(t), max(t), len(t)))

    # FAR
    if bad:
        leaked = [r for r in bad if r["resultado"] == "GRANTED"]
        print("\nFAR = %d / %d = %.1f %%" % (len(leaked), len(bad), percent(len(leaked), len(bad))))
        for r in leaked:
            print("   aceptada una clave incorrecta (secuencia leida: %s)" % r["secuencia"])
        if len(leaked) == 0 and len(bad) >= 10:
            print("   (con 0 errores en %d intentos, la cota superior al 95 %% es ~%.0f %%: regla de tres)"
                  % (len(bad), percent(3, len(bad))))
        results = {}
        for r in bad:
            results[r["resultado"]] = results.get(r["resultado"], 0) + 1
        print("   resultados de los incorrectos: " + ", ".join("%s %d" % (k, v) for k, v in sorted(results.items())))

    # FPS
    fps = read_rows(FPS_FILE)
    if fps:
        v = [float(r["fps_medios"]) for r in fps]
        ms = [float(r["ms_por_frame"]) for r in fps]
        print("\nFPS medios: %.1f (min %.1f, max %.1f, %d ejecuciones) | procesado medio %.1f ms/frame"
              % (sum(v) / len(v), min(v), max(v), len(v), sum(ms) / len(ms)))


if __name__ == "__main__":
    main()
