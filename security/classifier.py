# classifier.py - Clasificador k-NN con rechazo (Fase 4)
#
# fit(hu, geo, hog, y, bloque)  -> aprende las plantillas y calcula tau
# predict(hu, geo, hog)         -> (etiquetas, distancias); etiqueta "?" si la distancia > tau
# predict_one(hu, geo, hog)     -> lo mismo para un solo frame (la demo de la Fase 5)
# save(ruta) / Classifier.load(ruta)
#
# Las entradas son las que devuelve features.extract(): hu (7), geo (4), hog (1764).
# Con varias muestras: arrays (N, 7), (N, 4), (N, 1764).
#
# Distancia entre dos muestras:
#   d = d_hog / s_hog + w * d_geohu / s_geohu
#   - d_hog   : distancia euclidea entre los HOG (ya vienen normalizados por bloque)
#   - d_geohu : distancia euclidea entre [hu, geo] estandarizados (z-score, 11 valores)
#   - s_*     : mediana de las distancias entre muestras del entrenamiento; asi las dos
#               partes tienen la misma escala aunque una tenga 1764 valores y la otra 11
# feats="hog" o "geohu" usa solo una de las dos partes.
#
# Rechazo: se calcula la distancia media a los k vecinos mas cercanos; si supera tau -> "?".
# tau = percentil pct de esa misma distancia medida a los k vecinos de la MISMA letra en
# OTROS bloques (leave-one-block-out). Con frames consecutivos casi iguales, mirar los
# vecinos del mismo bloque daria un tau demasiado pequeno.

import numpy as np

UNKNOWN = "?"      # etiqueta que espera decoder.py: hay mano pero no es ningun signo


def euclid(a, b):
    """Matriz de distancias euclideas entre las filas de a (n, d) y las de b (m, d)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sq = np.sum(a ** 2, axis=1)[:, None] + np.sum(b ** 2, axis=1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0))


def geohu(hu, geo):
    """Une Hu (7) y geometria (4) en un vector de 11 valores."""
    return np.hstack([np.atleast_2d(hu), np.atleast_2d(geo)]).astype(np.float64)


class Classifier:
    def __init__(self, feats="comb", k=3, w=1.0, pct=95):
        self.feats = feats      # "hog", "geohu" o "comb"
        self.k = k
        self.w = w              # peso de geo+Hu frente a HOG (solo cuenta con feats="comb")
        self.pct = pct          # percentil de tau

    # ------------------------------------------------------------ distancias
    def _dist(self, hog, ghu_z):
        """Distancia de cada fila de test a cada plantilla: (n, n_train)."""
        d = np.zeros((len(hog), len(self.y)))
        if self.feats in ("hog", "comb"):
            d += euclid(hog, self.x_hog) / self.s_hog
        if self.feats == "geohu":
            d += euclid(ghu_z, self.x_ghu) / self.s_ghu
        elif self.feats == "comb":
            d += self.w * euclid(ghu_z, self.x_ghu) / self.s_ghu
        return d

    def _zscore(self, hu, geo):
        return (geohu(hu, geo) - self.mean) / self.std

    # ------------------------------------------------------------ entrenamiento
    def fit(self, hu, geo, hog, y, bloque=None):
        y = np.asarray(y)
        ghu = geohu(hu, geo)
        # el z-score se calcula SOLO con los datos de entrenamiento
        self.mean = ghu.mean(axis=0)
        self.std = ghu.std(axis=0) + 1e-6
        self.x_ghu = (ghu - self.mean) / self.std
        self.x_hog = np.asarray(hog, np.float64)
        self.y = y

        # escalas: mediana de las distancias entre muestras distintas del entrenamiento
        off = ~np.eye(len(y), dtype=bool)
        self.s_hog = np.median(euclid(self.x_hog, self.x_hog)[off])
        self.s_ghu = np.median(euclid(self.x_ghu, self.x_ghu)[off])

        # tau: distancia media a los k vecinos de la misma letra en otros bloques
        d = self._dist(self.x_hog, self.x_ghu)
        dists = []
        for i in range(len(y)):
            same = y == y[i]
            if bloque is None:
                same[i] = False                          # solo se excluye a si misma
            else:
                same &= (np.asarray(bloque) != bloque[i])  # y a todo su bloque
            nearest = np.sort(d[i, same])[:self.k]
            dists.append(nearest.mean())
        self.intra = np.array(dists)
        self.tau = np.percentile(self.intra, self.pct)
        return self

    # ------------------------------------------------------------ prediccion
    def predict(self, hu, geo, hog, reject=True):
        """Devuelve (etiquetas, distancia media a los k vecinos). Con reject=False nunca da "?"."""
        d = self._dist(np.atleast_2d(hog), self._zscore(hu, geo))
        labels, dists = [], []
        for row in d:
            idx = np.argsort(row)[:self.k]
            votes = {}
            for j in idx:                                # votos y distancia por letra
                votes.setdefault(self.y[j], []).append(row[j])
            # gana la letra con mas votos; si empatan, la de menor distancia media
            best = min(votes, key=lambda c: (-len(votes[c]), np.mean(votes[c])))
            dist = row[idx].mean()
            labels.append(UNKNOWN if reject and dist > self.tau else str(best))
            dists.append(dist)
        return np.array(labels), np.array(dists)

    def predict_one(self, hu, geo, hog, reject=True):
        labels, dists = self.predict(hu, geo, hog, reject)
        return str(labels[0]), float(dists[0])

    # ------------------------------------------------------------ guardar / cargar
    def save(self, path):
        np.savez(path, feats=self.feats, k=self.k, w=self.w, pct=self.pct, tau=self.tau,
                 mean=self.mean, std=self.std, s_hog=self.s_hog, s_ghu=self.s_ghu,
                 x_hog=self.x_hog, x_ghu=self.x_ghu, y=self.y, intra=self.intra)

    @classmethod
    def load(cls, path):
        m = np.load(path)
        c = cls(str(m["feats"]), int(m["k"]), float(m["w"]), float(m["pct"]))
        c.tau = float(m["tau"])
        c.mean, c.std = m["mean"], m["std"]
        c.s_hog, c.s_ghu = float(m["s_hog"]), float(m["s_ghu"])
        c.x_hog, c.x_ghu, c.y, c.intra = m["x_hog"], m["x_ghu"], m["y"], m["intra"]
        return c
