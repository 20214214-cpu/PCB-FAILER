# PCB Defect Detection - Clasificación Multiclase

Sistema de detección automática de defectos en PCB usando redes convolucionales y clasificación multiclase.

## 🎯 Clases Detectadas

El sistema trabaja actualmente con **5 clases**:

1. **ok** – PCB sin defectos
2. **Missing_hole** – Agujeros faltantes
3. **Open_circuit** – Circuito abierto
4. **Short** – Cortocircuito
5. **Spur** – Espuelas/protuberancias

## 📁 Estructura del Dataset

```
pcb-defects/
├── images/
│   ├── Missing_hole/
│   ├── Open_circuit/
│   ├── Short/
│   └── Spur/
└── PCB_USED/              # Imágenes sin defectos (clase ok)
```

## 🚀 Uso

### 1. Entrenamiento del Modelo (ResNet-18)

```bash
python main_multiclass.py
```

El script de entrenamiento:
- Construye el dataset con división estratificada 70/15/15.
- Aplica data augmentation específico para PCB (resize 400×400, flips, rotación ±5°, color jitter leve).
- Entrena una **ResNet-18** preentrenada adaptada a 5 clases.
- Usa `WeightedRandomSampler` y pesos en la pérdida para manejar desbalance.
- Implementa early stopping con paciencia 10 y `MIN_DELTA = 1e-4`.
- Reduce el learning rate ×0.5 cuando no hay mejora en validación durante 3 épocas consecutivas.
- Guarda el mejor modelo como `pcb_resnet18_multiclass.pth`.
- Genera gráficas de loss/accuracy, matriz de confusión y curva ROC.

**Parámetros clave (ver `main_multiclass.py`):**
- `EPOCHS = 50`
- `BATCH_SIZE = 16`
- `LR = 1e-5`
- `EARLY_STOPPING_PATIENCE = 10`
- `LR_REDUCE_PATIENCE = 3`
- `LR_REDUCE_FACTOR = 0.5`
- `MIN_DELTA = 1e-4`
- `OK_REPLICATION_FACTOR = 1.05` (si `BALANCE_OK_CLASS = True`)

### 2. Inferencia en Imágenes (ResNet-50)

```bash
python infer_multiclass.py --image ruta/a/imagen.png
python infer_multiclass.py --batch img1.png img2.png
python infer_multiclass.py --image test.png --no-plot
```

El script de inferencia carga el modelo `pcb_resnet50_multiclass.pth`, ajusta las entradas a 512×512 y produce una visualización con la confianza por clase (opcionalmente guardada como `prediction_{class}_{name}.png`).

**Ejemplo de salida:**
```
Analizando imagen: test_pcb.png

==================================================
RESULTADO:
  Clase predicha: Missing_hole
  Confianza: 94.23%

Probabilidades completas:
  ok                  :   2.15%
  Missing_hole        :  94.23%
  Open_circuit        :   0.87%
  Short               :   0.65%
  Spur                :   2.10%
==================================================
```

### 3. Detección en Tiempo Real (Webcam)

```bash
python infer_realtime.py              # Cámara predeterminada
python infer_realtime.py --camera 1   # Selecciona cámara
python infer_realtime.py --list       # Lista cámaras disponibles
python infer_realtime.py --camera 0 --width 1920 --height 1080
```

Controles durante la ejecución: `q` (salir), `s` (captura), `c` (cambiar cámara), `SPACE` (pausa).

## 📊 Outputs Generados

Durante entrenamiento:
- `pcb_resnet18_multiclass.pth`
- `training_history.png`
- `confusion_matrix_multiclass.png`
- `roc_curve_multiclass.png`
- `pcb_model_graph_multiclass.png`

Durante inferencia:
- `prediction_{class}_{name}.png`

Durante detección en tiempo real:
- `capture_{n}_{class}.png`

## 📐 Métricas Clave y Teoría

Todas las métricas se derivan de la **matriz de confusión**, que contabiliza verdaderos positivos (TP), falsos positivos (FP), verdaderos negativos (TN) y falsos negativos (FN) por clase. A partir de ella se calculan:

- **Accuracy**: Proporción de predicciones correctas sobre el total.
  - Fórmula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
  - Útil como medida global, pero puede sesgarse si las clases están desbalanceadas.

- **Precision**: Qué porcentaje de las predicciones positivas son correctas.
  - Fórmula: `Precision = TP / (TP + FP)`
  - Alta precisión implica pocos falsos positivos.

- **Recall (Sensibilidad)**: Qué porcentaje de los ejemplos positivos reales se detectan.
  - Fórmula: `Recall = TP / (TP + FN)`
  - Alta sensibilidad implica pocos falsos negativos, clave para no omitir defectos.

- **F1-Score**: Media armónica entre precision y recall.
  - Fórmula: `F1 = 2 * Precision * Recall / (Precision + Recall)`
  - Equilibra ambos indicadores; útil cuando se requiere balancear FP y FN.

- **ROC (Receiver Operating Characteristic)**: Curva que grafica la tasa de verdaderos positivos (TPR) frente a la tasa de falsos positivos (FPR) al variar el umbral de decisión.
  - `TPR = TP / (TP + FN)`, `FPR = FP / (FP + TN)`
  - En multiclase se calcula una curva por clase usando estrategia one-vs-all.

- **AUC (Area Under the Curve)**: Área bajo la curva ROC.
  - Valor entre 0 y 1; cuanto más cercano a 1, mejor es la separabilidad entre clases.

La combinación de estas métricas permite evaluar no solo la tasa global de aciertos, sino también cómo se comporta el modelo ante cada tipo de defecto. El proyecto genera reportes con precision, recall y F1 por clase, además de la matriz de confusión y curvas ROC para análisis visual.

## 🔧 Requisitos

```bash
pip install torch torchvision
pip install pillow numpy matplotlib seaborn
pip install scikit-learn tqdm
pip install torchviz graphviz
pip install opencv-python            # Para tiempo real
```

## 📈 Mejoras Implementadas

- ✅ Clasificación multiclase (5 clases) en lugar de binaria.
- ✅ Balanceo mediante sampler ponderado, pesos en la loss y réplica opcional de clase OK.
- ✅ Early stopping con paciencia extendida y reducción dinámica de LR.
- ✅ Data augmentation ajustado a PCB (400×400, rotaciones suaves, flips, ajustes leves de color).
- ✅ Visualizaciones de entrenamiento, matriz de confusión y curvas ROC.
- ✅ Script de inferencia dedicado con visualización clara de probabilidades.
- ✅ Flujo de detección en tiempo real con OpenCV y controles interactivos.

## 🎓 Modelos

- **Entrenamiento:** ResNet-18 preentrenada (entrada 400×400). Optimización con Adam (`lr=1e-5`), CrossEntropyLoss con pesos por clase y refuerzo opcional para "ok".
- **Inferencia:** ResNet-50 finetuneada para 5 clases (`pcb_resnet50_multiclass.pth`), entrada 512×512, usada en `infer_multiclass.py` e `infer_realtime.py`.

## ⚙️ Configuración de Balanceo

```python
BALANCE_OK_CLASS = False
OK_REPLICATION_FACTOR = 1.05
```

- Usa `BALANCE_OK_CLASS = True` para replicar ligeramente la clase "ok" según `OK_REPLICATION_FACTOR`.
- Ajusta el factor en función del desbalance real; combínalo con los pesos automáticos de clase ya integrados en la pérdida y el sampler.

## 📝 Notas

- El entrenamiento se detiene si no hay mejora > `MIN_DELTA = 1e-4` durante `EARLY_STOPPING_PATIENCE = 10` épocas.
- El learning rate se reduce ×0.5 tras 3 épocas sin mejora de accuracy en validación.
- Se recomienda disponer de al menos 50–100 imágenes por clase.
- La división estratificada asegura proporciones consistentes en train/val/test.

## 🆚 Comparación con Versión Binaria

| Característica | Binaria (`main_singlesclass.py`) | Multiclase (`main_multiclass.py`) |
|----------------|----------------------------------|-----------------------------------|
| Clases         | 2 (ok / defectuoso)              | 5 (ok + 4 defectos específicos)   |
| Objetivo       | Detección general de defectos    | Identificación del tipo de defecto |
| Complejidad    | Menor                             | Mayor, requiere más datos         |
| Modelo         | ResNet-18                         | ResNet-18 / ResNet-50             |

---
 
**Desarrollado para automatizar la detección de defectos en PCB** 🔍⚡
