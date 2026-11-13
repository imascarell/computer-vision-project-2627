# Plan de trabajo — Sistema de seguridad

Proyecto final de **Visión por Ordenador I** (iMAT, ICAI · curso 2026/27).

## Resumen del sistema

El usuario se identifica introduciendo ante la cámara una **secuencia de 5 signos estáticos del alfabeto dactilológico de la LSE**: la palabra clave **C-U-B-I-L**. Si la secuencia es correcta y está en orden, se desbloquea el sistema propuesto.

- **Visión clásica + ML clásico** (sin deep learning): segmentación por color + features geométricas/HOG + k-NN con rechazo.
- El reconocedor solo conoce los **5 signos registrados** (C, U, B, I, L); cualquier otra cosa se clasifica como *desconocido*.
- El decodificador es una **máquina de estados** de dos niveles (clave S0→S5 y confirmación de cada signo).

```
Cámara → Undistort → ROI → YCrCb → Máscara (piel ∧ resta de fondo) → Morfología
      → Contorno mayor → Máscara 64×64 → Features (Hu, geometría, HOG)
      → k-NN con rechazo → etiqueta por frame → Decodificador (máquina de estados)
      → ACCESO CONCEDIDO / DENEGADO
```

---

## Fases

### Fase 0 — Propuesta
- Redactar el bloque de seguridad: palabra clave CUBIL, pipeline de reconocimiento y máquina de estados.
- Diagrama de bloques del sistema.

**Hecho cuando:** la propuesta está entregada en Moodle.

### Fase 1 — Base
- Estructura del repositorio y README.
- Lectura de vídeo desde la cámara con **FPS sobreimpresos**.
- **Calibración offline** con patrón (reutilizando la Práctica 2) → `calib.npz` con matriz intrínseca K, coeficientes de distorsión y **RMS**.
- Precálculo de los mapas de corrección de distorsión (`initUndistortRectifyMap`) una sola vez y `remap` por frame.

**Hecho cuando:** se ve el vídeo en directo corregido con FPS ≥ 25.

### Fase 2 — Segmentación de la mano
- ROI fija en pantalla donde se coloca la mano.
- Conversión a **YCrCb** y umbral de piel en Cr/Cb.
- Resta con un **fondo de referencia** capturado al inicio (ROI vacía) y combinación AND con la máscara de piel.
- Morfología (apertura + cierre), selección del **contorno mayor** y relleno.
- Recorte por bounding box y normalización a **máscara binaria 64×64**.
- Ventanas de depuración de cada paso y guardado de capturas.

**Hecho cuando:** las 5 letras dan una máscara limpia en el escenario de trabajo y están guardadas las capturas de cada paso (= *secuencia de transformaciones* del informe).

### Fase 3 — Features y registro de plantillas
- Features geométricas: momentos de **Hu** (log), solidez, relación de aspecto, extensión, nº de **defectos de convexidad**.
- **HOG** sobre la máscara 64×64.
- Script de registro: ~40 frames por letra (con variación de posición y giro) → `dataset.npz`.

**Hecho cuando:** existe el dataset de C, U, B, I, L grabado en las condiciones de la demo.

### Fase 4 — Clasificador (go / no-go)
- **k-NN** (k = 3–5) con **rechazo**: si la distancia media a los vecinos supera τ → *desconocido* (τ = percentil 95 de las distancias intra-clase).
- Evaluación offline con grabaciones de **otro día y otra iluminación**: matriz de confusión.
- Comparativa opcional con SVM para el informe.

**Hecho cuando:** precisión **> 90 % por letra**, con **U y B** bien separadas.
**Si no se cumple:** ajustar features o cambiar la palabra clave a **CLIVO**.

### Fase 5 — Integración
- Bucle principal: cámara → reconocedor → `decoder.py` → overlay.
- Teclas: `r` (reset de la secuencia), `n` (modo administrador: nueva clave), `q` (salir).
- Ajuste en vivo de los parámetros del decodificador: ventana N, ratio de mayoría, frames de neutro M y timeout.

**Hecho cuando:** CUBIL desbloquea en directo y cualquier secuencia incorrecta se deniega.

### Fase 6 — Pruebas y métricas
- 20 intentos con la clave correcta y 20 con claves incorrectas.
- Métricas: **FRR** (falso rechazo), **FAR** (falsa aceptación), tiempo medio de introducción de la clave y **FPS medios**.
- Pruebas de robustez con otra luz y otro fondo; optimización si bajan los FPS.

**Hecho cuando:** está la tabla de resultados para el informe.

### Fase 7 — Documentación
- Figuras: diagrama de bloques, secuencia de transformaciones de la imagen y diagrama de estados.
- Sección de resultados (matriz de confusión, FRR/FAR, FPS).
- Sección del sistema de seguridad en el README.
- Grabación del tramo de desbloqueo para el vídeo demo.

**Hecho cuando:** la sección *Sistema de seguridad* del informe está cerrada.

---

## Decodificador (máquina de estados)

**Nivel clave:** `INPUT (S0…S4) → verificar en S5 → GRANTED / DENIED`
- Cada signo confirmado avanza un estado; si no coincide con la clave se activa un flag de error, pero se sigue avanzando (no se revela en qué posición se falló).
- `DENIED` × 3 seguidos → `LOCKED` (bloqueo temporal).
- `ENROLL` (modo administrador): graba una nueva clave de 5 signos distintos.
- Reset: tecla `r` o timeout sin confirmar signo.

**Nivel signo:** `WAIT_HAND → DETECTING → WAIT_NEUTRAL`
- Un signo se confirma si ocupa ≥ 80 % de los últimos N frames (y no es *desconocido*).
- Hay que retirar la mano M frames antes del siguiente signo (evita contar dos veces el mismo).

---

## Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| Iluminación variable | Umbral solo en Cr/Cb; exposición y balance de blancos fijos; recalibrar piel al arrancar |
| Fondo con tonos de piel | ROI fija + fondo liso + AND con resta de fondo |
| Cara / brazo en la máscara | ROI que excluye la cara; manga larga; contorno mayor |
| Escala, rotación, posición | Normalización a 64×64; Hu invariante; registro con variación |
| Confusión U / B | Relación de aspecto y área + HOG; vigilar en la matriz de confusión |
| Falsos positivos en transiciones | Ventana de N frames + neutro obligatorio + rechazo |

**Escenario fijo** desde la fase 2: mismo fondo liso, misma luz y manga larga para el dataset y la demo.
