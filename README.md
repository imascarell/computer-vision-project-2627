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

Patrón: tablero de **10×7 cuadros de 25 mm**, es decir, **9×6 esquinas interiores** (`calibration/pattern/patron_10x7_25mm.pdf`).

- Imprimir al 100 % (tamaño real, sin "ajustar a página").
- Pegarlo sobre un cartón rígido.
- Medir un cuadro con una regla y usar esa medida en la calibración.

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
