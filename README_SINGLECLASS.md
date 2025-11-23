# PCB Defect Detection - Clasificación Binaria

Clasificador binario para detectar si una PCB está libre de defectos (`ok`) o es defectuosa (`defective`) empleando fine-tuning de **ResNet-50**.

## 🎯 Clases Detectadas

- **ok** – PCB sin defectos
- **defective** – PCB con cualquier tipo de defecto (Missing hole, Open circuit, Short, Spur, etc.)

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

## 🚀 Ejecución del Entrenamiento

```bash
python main_singlesclass.py
```

El script realiza los siguientes pasos:
- Construye la lista de imágenes etiquetando `PCB_USED` como clase `ok` y el resto como `defective`.
- Duplica las muestras `ok` para reducir el desbalance frente a la clase defectuosa.
- Divide el dataset en `70 %` entrenamiento, `15 %` validación y `15 %` test de manera estratificada.
- Aplica data augmentation ligero (resize 400×400 + `ColorJitter`) y normalización imagenet.
- Ajusta una **ResNet-50** preentrenada a 2 clases (`ok`, `defective`).
- Entrena durante `12` épocas con **Adam** (`lr = 1e-4`).
- Guarda el mejor modelo (según accuracy de validación) como `pcb_resnet50.pth`.
- Evalúa sobre el conjunto de test generando métricas, curva ROC y matriz de confusión.

## 🔧 Parámetros Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `NUM_CLASSES` | `2` | Clases binarias: ok / defective |
| `BATCH_SIZE` | `16` | Tamaño de lote |
| `EPOCHS` | `12` | Épocas de entrenamiento |
| `LR` | `1e-4` | Learning rate para Adam |
| `OUT_MODEL` | `pcb_resnet50.pth` | Ruta del modelo guardado |
| `MY_IMAGE`, `MY_IMAGE2` | Imagen(es) para inferencia rápida |

## 📊 Métricas y Salidas

Durante el entrenamiento y evaluación el script genera:
- `roc_curve_binary.png` – Curva ROC y AUC.
- `confusion_matrix.png` – Matriz de confusión binaria.
- Reporte de clasificación (accuracy, precision, recall, F1) impreso en consola.
- Probabilidades y etiquetas predichas para `MY_IMAGE` y `MY_IMAGE2` (si existen).
- `pcb_model_graph.png` – Grafo de la arquitectura exportado con TorchViz.

### Definiciones Rápidas

Las métricas se calculan a partir de la matriz de confusión (TP, FP, TN, FN):
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision**: `TP / (TP + FP)` – Fracción de predicciones positivas correctas.
- **Recall** (sensibilidad): `TP / (TP + FN)` – Cobertura de la clase positiva (defectiva).
- **F1-score**: `2 * Precision * Recall / (Precision + Recall)` – Equilibra precision y recall.
- **ROC / AUC**: relación entre `TPR = TP / (TP + FN)` y `FPR = FP / (FP + TN)` para distintos umbrales; el área bajo la curva resume la separabilidad global.

## 🧪 Inferencia Rápida

Puedes colocar imágenes en la raíz del proyecto y actualizar `MY_IMAGE` / `MY_IMAGE2`. Al finalizar el entrenamiento (o cargando el modelo guardado) el script imprimirá la etiqueta estimada y las probabilidades correspondientes.

## 📦 Dependencias Requeridas

```bash
pip install torch torchvision
pip install pillow numpy matplotlib
pip install scikit-learn tqdm
pip install torchviz graphviz
```

## 📝 Notas y Recomendaciones

- Asegúrate de que el dataset tenga suficientes ejemplos para ambas clases; si la clase `ok` es minoritaria puedes aumentar manualmente su replicación en `build_image_list`.
- Verifica que `graphviz` esté instalado en el sistema operativo para exportar correctamente `pcb_model_graph.png`.
- Ajusta `EPOCHS`, `LR` o los parámetros de data augmentation según el rendimiento observado.
- Si dispones de GPU, PyTorch la utilizará automáticamente (`DEVICE = cuda`).

---

**Desarrollado para validación rápida binaria de defectos en PCB** 🔍⚡
