# Proyecto final — Visión por Ordenador I

iMAT, ICAI · curso 2026/27 · Grupo B

Sistema de visión en tiempo real formado por dos bloques:

1. **Sistema de seguridad**: el usuario desbloquea el sistema haciendo ante la cámara una secuencia de 5 signos estáticos del alfabeto dactilológico de la LSE (clave **C-U-B-I-L**). Se usa visión clásica y ML clásico (segmentación por color, features geométricas y HOG, k-NN con rechazo), sin deep learning.
2. **Sistema propuesto**: se ejecuta una vez concedido el acceso (en desarrollo).

El plan detallado del sistema de seguridad está en [`docs/PLAN_SISTEMA_SEGURIDAD.md`](docs/PLAN_SISTEMA_SEGURIDAD.md).

## Estructura

```
.
├── calibration/            # Calibración de la cámara (offline)
│   ├── pattern/            # Patrón de calibración (PDF para imprimir)
│   └── images/             # Fotos del patrón tomadas con la cámara de la demo
├── security/               # Sistema de seguridad
│   ├── decoder.py          # Máquina de estados que decodifica la secuencia de signos
│   ├── test_decoder.py     # Prueba del decodificador con etiquetas simuladas
│   ├── data/               # Dataset de plantillas de las letras
│   └── captures/           # Capturas de cada paso del procesado (para el informe)
├── proposed_system/        # Sistema propuesto
├── docs/                   # Plan de trabajo y documentación
└── requirements.txt
```

## Instalación

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

## Uso

### Cámara con FPS

```bash
cd security
python camera.py
```

Abre la cámara a 640×480 y muestra los FPS en pantalla. Se cierra con `q`.

### Prueba del decodificador (sin cámara)

```bash
cd security
python test_decoder.py
```

## Calibración de la cámara

Se usa la cámara del portátil a una resolución fija. La calibración se hace con la **misma cámara y la misma resolución** que la demo.

Patrón impreso: tablero de **7×9 cuadros**, es decir, **6×8 esquinas interiores** (`PATTERN = (6, 8)`); es el que detecta OpenCV con la cámara. El PDF de `calibration/pattern/` es de 10×7 cuadros de 25 mm (9×6 esquinas), pero la copia impresa no coincide con él. El tamaño del cuadro se mide con regla sobre el papel impreso.

- Imprimir al 100 % (tamaño real, sin "ajustar a página").
- Pegarlo sobre un cartón rígido.
- Medir un cuadro con una regla y usar esa medida en la calibración.

La imagen no se voltea en ningún punto del proyecto: calibración, dataset y demo usan el flujo de la cámara tal como llega.

### Calibración

```bash
cd calibration
python calibrate.py
```

Calcula K, la distorsión y el RMS con las imágenes de `calibration/images/` y guarda `calibration/calib.npz`.
Resultado actual (20 imágenes, 640×480, cuadro de 25 mm): **RMS = 0,112 px**, fx ≈ 579, fy ≈ 577, cx ≈ 316, cy ≈ 256, distorsión radial muy pequeña (k1 ≈ 0,003; k2 ≈ −0,035).
`cornerSubPix` usa una ventana de 5 px porque los cuadros miden unos 12 px en la imagen; con 11 px el RMS sube a ≈ 5 px.

### Captura de imágenes

```bash
cd calibration
python capture.py
```

Con `s` se guarda la imagen (solo si se detecta el tablero) en `calibration/images/`; con `q` se sale. Hacen falta unas 15–20 imágenes variando distancia, inclinación y posición del tablero, incluyendo los bordes y las esquinas del encuadre.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Propuesta | Hecha |
| 1 | Base: cámara, FPS, calibración y undistort | En curso |
| 2 | Segmentación de la mano | Pendiente |
| 3 | Features y registro de plantillas | Pendiente |
| 4 | Clasificador k-NN | Pendiente |
| 5 | Integración | Pendiente |
| 6 | Pruebas y métricas | Pendiente |
| 7 | Documentación | Pendiente |
