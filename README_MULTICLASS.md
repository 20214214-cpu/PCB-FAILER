# PCB Defect Detection - Clasificación Multiclase

Sistema de detección automática de defectos en PCB usando Deep Learning con clasificación multiclase.

## 🎯 Clases Detectadas

El sistema identifica **6 clases** diferentes:

1. **ok** - PCB sin defectos
2. **Missing_hole** - Agujeros faltantes
3. **Open_circuit** - Circuito abierto
4. **Short** - Cortocircuito
5. **Spur** - Espuelas/protuberancias

## 📁 Estructura del Dataset

```
pcb-defects/
├── images/
│   ├── Missing_hole/
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
- Carga las imágenes del dataset (replica clase OK si `BALANCE_OK_CLASS=True`)
- Entrena un modelo ResNet18 con 6 clases
- Usa data augmentation específico para PCB (rotación ±5°, flips, blur suave)
- Implementa early stopping (detiene si no mejora en 10 épocas)
- Guarda el mejor modelo como `pcb_resnet18_multiclass.pth`
- Genera gráficas de entrenamiento y matriz de confusión

**Configuración importante:**
- `EPOCHS = 50` - Número máximo de épocas de entrenamiento
- `EARLY_STOPPING_PATIENCE = 10` - Detiene si no mejora en 10 épocas
- `MIN_DELTA = 0.001` - Mejora mínima requerida para continuar
- `BATCH_SIZE = 16` - Tamaño del lote
- `LR = 1e-4` - Tasa de aprendizaje
- `BALANCE_OK_CLASS = True` - Activa/desactiva replicación de clase OK
- `OK_REPLICATION_FACTOR = 3` - Factor de replicación para balanceo

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

✅ **Clasificación multiclase** - 6 clases en lugar de binario  
✅ **Balanceo automático de clases** - Replica imágenes OK 3x (configurable)  
✅ **Early Stopping** - Detección automática de convergencia  
✅ **Data augmentation optimizado para PCB**:
  - Rotación suave ±5° (sin deformar componentes)
  - Flips horizontal y vertical
  - Ajustes de brillo/contraste moderados (15%)
  - Gaussian blur suave ocasional
  - Sin saturación ni deformaciones agresivas
✅ **Entrenamiento extendido** - Hasta 50 épocas con early stopping  
✅ **Visualizaciones mejoradas** - Matriz de confusión con seaborn  
✅ **Script de inferencia dedicado** - Fácil uso en producción  
✅ **Detección en tiempo real** - Usando cámara web con OpenCV  
✅ **Selector de cámara interactivo** - Cambio dinámico de fuente  
✅ **Métricas detalladas** - Classification report por clase  

## 🎓 Modelo

- **Arquitectura**: ResNet18 (pre-entrenado en ImageNet)
- **Fine-tuning**: Última capa adaptada a 6 clases
- **Input size**: 224x224 RGB
- **Optimizador**: Adam (lr=1e-4)
- **Loss**: CrossEntropyLoss con pesos por clase
- **Early Stopping**: Patience=10, Min Delta=0.001
- **Entrenamiento**: Hasta 50 épocas con detección automática de convergencia

## ⚙️ Configuración de Balanceo

El sistema incluye balanceo automático de la clase "OK" para compensar el desbalance entre PCBs correctos y defectuosos:

```python
BALANCE_OK_CLASS = True          # Activar/desactivar balanceo
OK_REPLICATION_FACTOR = 3        # Replicar imágenes OK 3x
```

**¿Por qué es importante?**
- Los datasets de PCB suelen tener pocas imágenes "OK" vs muchas con defectos
- Sin balanceo, el modelo puede sesgar hacia detectar defectos
- La replicación 3x mejora la detección de PCBs correctos sin afectar precisión en defectos

**Cómo ajustar:**
- `BALANCE_OK_CLASS = False` → Sin replicación (usar dataset original)
- `OK_REPLICATION_FACTOR = 2` → Duplicar imágenes OK
- `OK_REPLICATION_FACTOR = 5` → Replicar 5 veces (para datasets muy desbalanceados)

## 📝 Notas

### Data Augmentation para PCB
El sistema usa aumentaciones **específicamente diseñadas para PCBs**:
- **Rotación limitada a ±5°**: Evita deformar componentes y trazas críticas
- **Sin saturación**: Los PCBs tienen colores estandarizados (verde, cobre)
- **Blur suave**: Simula variaciones de enfoque sin perder detalles
- **Brightness/Contrast moderado**: Simula diferentes condiciones de iluminación

❌ **No usar**: Crop agresivo, deformaciones, saturación alta, rotaciones >10°

### Early Stopping
El entrenamiento se detiene automáticamente cuando:
- No hay mejora en validation accuracy por `EARLY_STOPPING_PATIENCE` (10) épocas consecutivas
- La mejora es menor a `MIN_DELTA` (0.001)
- Esto previene overfitting y ahorra tiempo de entrenamiento

### Balanceo de Clases
Se usan **dos estrategias complementarias**:
1. **Replicación de datos** (opcional): Multiplica imágenes OK por `OK_REPLICATION_FACTOR`
2. **Class weights automáticos**: Ajusta la loss function según frecuencia de cada clase

### Recomendaciones
- Se recomienda tener al menos **50-100 imágenes por clase** para buenos resultados
- El modelo guardado (`pcb_resnet18_multiclass.pth`) es el de **mejor validation accuracy**
- La división es **70% train, 15% validation, 15% test** con seed fijo para reproducibilidad

## 🆚 Comparación con Versión Binaria

| Característica | Binaria (`main.py`) | Multiclase (`main_multiclass.py`) |
|----------------|---------------------|-------------------------------------|
| Clases         | 2 (ok/defective)    | 6 (ok + 5 tipos de defectos)       |
| Precisión      | Alta para detectar defectos | Identifica tipo específico     |
| Uso            | Screening inicial   | Diagnóstico detallado              |
| Entrenamiento  | Más rápido          | Requiere más datos                 |

---

**Desarrollado para detección automática de defectos en PCB** 🔍⚡
