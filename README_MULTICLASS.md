# PCB Defect Detection - Clasificación Multiclase

Sistema de detección automática de defectos en PCB usando Deep Learning con clasificación multiclase.

## 🎯 Clases Detectadas

El sistema identifica **6 clases** diferentes:

1. **ok** - PCB sin defectos
2. **Missing_hole** - Agujeros faltantes
3. **Mouse_bite** - Mordeduras (defecto en bordes)
4. **Open_circuit** - Circuito abierto
5. **Short** - Cortocircuito
6. **Spur** - Espuelas/protuberancias

## 📁 Estructura del Dataset

```
pcb-defects/
├── images/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   └── Spur/
└── PCB_USED/  (imágenes sin defectos)
```

## 🚀 Uso

### 1. Entrenamiento del Modelo

```bash
python main_multiclass.py
```

Este script:
- Carga las imágenes del dataset
- Entrena un modelo ResNet18 con 7 clases
- Guarda el mejor modelo como `pcb_resnet18_multiclass.pth`
- Genera gráficas de entrenamiento y matriz de confusión

**Configuración importante:**
- `EPOCHS = 15` - Número de épocas de entrenamiento
- `BATCH_SIZE = 16` - Tamaño del lote
- `LR = 1e-4` - Tasa de aprendizaje

### 2. Inferencia en Imágenes

#### Imagen individual:
```bash
python infer_multiclass.py --image ruta/a/imagen.png
```

#### Múltiples imágenes:
```bash
python infer_multiclass.py --batch imagen1.png imagen2.png imagen3.png
```

#### Sin visualización:
```bash
python infer_multiclass.py --image test.png --no-plot
```

### 3. Detección en Tiempo Real (Cámara Web)

```bash
# Usar cámara predeterminada (0)
python infer_realtime.py

# Especificar cámara
python infer_realtime.py --camera 1

# Listar cámaras disponibles
python infer_realtime.py --list

# Con resolución personalizada
python infer_realtime.py --camera 0 --width 1920 --height 1080
```

**Controles durante la ejecución:**
- `q` - Salir
- `s` - Guardar captura de pantalla
- `c` - Cambiar cámara
- `SPACE` - Pausar/Reanudar

**Características:**
- ✅ Predicción en tiempo real con FPS
- ✅ Visualización de probabilidades por clase
- ✅ Colores distintivos para cada tipo de defecto
- ✅ Captura de pantallas con nombre automático
- ✅ Selector de cámara en vivo
- ✅ Interfaz visual optimizada

### 4. Ejemplo de Salida

```
Analizando imagen: test_pcb.png

==================================================
RESULTADO:
  Clase predicha: Missing_hole
  Confianza: 94.23%

Probabilidades completas:
  ok                  :   2.15%
  Missing_hole        :  94.23%
  Mouse_bite          :   1.45%
  Open_circuit        :   0.87%
  Short               :   0.65%
  Spur                :   0.42%
==================================================
```

## 📊 Outputs Generados

Durante el entrenamiento:
- `pcb_resnet18_multiclass.pth` - Modelo entrenado
- `training_history.png` - Gráficas de loss y accuracy
- `confusion_matrix_multiclass.png` - Matriz de confusión
- `pcb_model_graph_multiclass.png` - Arquitectura del modelo

Durante la inferencia:
- `prediction_{class}_{name}.png` - Visualización de la predicción

Durante detección en tiempo real:
- `capture_{n}_{class}.png` - Capturas guardadas con `s`

## 🔧 Requisitos

```bash
pip install torch torchvision
pip install pillow numpy matplotlib seaborn
pip install scikit-learn tqdm
pip install torchviz graphviz
pip install opencv-python  # Para detección en tiempo real
```

## 📈 Mejoras Implementadas

✅ **Clasificación multiclase** - 7 clases en lugar de binario  
✅ **Balanceo de clases** - Pesos automáticos en la función de pérdida  
✅ **Data augmentation** - Rotaciones, flips, color jitter  
✅ **Visualizaciones mejoradas** - Matriz de confusión con seaborn  
✅ **Script de inferencia dedicado** - Fácil uso en producción  
✅ **Detección en tiempo real** - Usando cámara web con OpenCV  
✅ **Selector de cámara interactivo** - Cambio dinámico de fuente  
✅ **Métricas detalladas** - Classification report por clase  

## 🎓 Modelo

- **Arquitectura**: ResNet18 (pre-entrenado en ImageNet)
- **Fine-tuning**: Última capa adaptada a 7 clases
- **Input size**: 224x224 RGB
- **Optimizador**: Adam
- **Loss**: CrossEntropyLoss con pesos por clase

## 📝 Notas

- El modelo usa **class weights** para manejar el desbalance entre clases
- Se recomienda tener al menos 50-100 imágenes por clase para buenos resultados
- El data augmentation ayuda a mejorar la generalización
- La división es 70% train, 15% validation, 15% test

## 🆚 Comparación con Versión Binaria

| Característica | Binaria (`main.py`) | Multiclase (`main_multiclass.py`) |
|----------------|---------------------|-------------------------------------|
| Clases         | 2 (ok/defective)    | 6 (ok + 5 tipos de defectos)       |
| Precisión      | Alta para detectar defectos | Identifica tipo específico     |
| Uso            | Screening inicial   | Diagnóstico detallado              |
| Entrenamiento  | Más rápido          | Requiere más datos                 |

---

**Desarrollado para detección automática de defectos en PCB** 🔍⚡
