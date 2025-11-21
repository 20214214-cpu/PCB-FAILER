import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_DIR = "pcb-defects"
MODEL_PATH = "pcb_resnet50_multiclass.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
SEED = 42

# Configuración de Clases (Actualizada sin Mouse_bite)
CLASS_NAMES = [
    "ok",                  # 0
    "Missing_hole",        # 1
    "Open_circuit",        # 2
    "Short",               # 3
    "Spur"                 # 4
]

DEFECT_FOLDERS = {
    "Missing_hole": 1,
    "Open_circuit": 2,
    "Short": 3,
    "Spur": 4
}

NUM_CLASSES = len(CLASS_NAMES)

# ==========================================
# PREPARACIÓN DE DATOS
# ==========================================
def build_image_list(data_dir):
    """Construye la lista de imágenes y etiquetas (sin replicación para test real)"""
    img_paths = []
    labels = []
    
    print("Escaneando directorio de datos...")
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = Path(root) / f
                
                if "PCB_USED" in root:
                    lbl = 0
                    img_paths.append(str(p))
                    labels.append(lbl)
                else:
                    # Ignorar Mouse_bite si quedó alguna carpeta
                    if "Mouse_bite" in root:
                        continue
                        
                    lbl = 0
                    found = False
                    for defect_name, defect_label in DEFECT_FOLDERS.items():
                        if defect_name in root:
                            lbl = defect_label
                            found = True
                            break
                    
                    if found:
                        img_paths.append(str(p))
                        labels.append(lbl)
    
    return img_paths, labels

class PCBClassDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Transformaciones (Solo normalización para test)
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# FUNCIONES DE EVALUACIÓN
# ==========================================
def evaluate_dummy(X_test, y_test, strategy="most_frequent"):
    """Evalúa un clasificador 'tonto' como línea base"""
    print(f"\n--- Evaluando Baseline Dummy ({strategy}) ---")
    dummy = DummyClassifier(strategy=strategy, random_state=SEED)
    dummy.fit(X_test, y_test) # El fit es simbólico en dummy
    
    start_time = time.time()
    preds = dummy.predict(X_test)
    inference_time = time.time() - start_time
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"Tiempo total inferencia (simulada): {inference_time:.4f}s")
    return acc, preds

def evaluate_model(model, dataloader):
    """Evalúa el modelo de Deep Learning"""
    print(f"\n--- Evaluando Modelo resnet50 ({MODEL_PATH}) ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc="Inferencia"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labs.numpy())
            
    inference_time = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Tiempo total inferencia: {inference_time:.4f}s")
    print(f"Promedio por imagen: {inference_time/len(all_labels)*1000:.2f}ms")
    
    return acc, all_preds, all_labels

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"=== BASELINE TESTER MULTICLASS ({NUM_CLASSES} clases) ===")
    
    # 1. Cargar Datos
    img_paths, labels = build_image_list(DATA_DIR)
    print(f"Total imágenes encontradas: {len(img_paths)}")
    
    # 2. Recrear Split (Misma semilla que entrenamiento para aislar Test set)
    indices = np.arange(len(labels))
    # Split 1: Train vs Rest
    _, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=SEED)
    # Split 2: Val vs Test
    temp_labels = [labels[i] for i in temp_idx]
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)
    
    test_paths = [img_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Imágenes en Test Set: {len(test_paths)}")
    
    # Dataset y Loader
    test_dataset = PCBClassDataset(test_paths, test_labels, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ---------------------------------------------------------
    # BASELINE 1: Dummy Classifier (Most Frequent)
    # ---------------------------------------------------------
    # Usamos features dummy solo para pasar al método, lo importante son los labels
    dummy_features = np.zeros((len(test_labels), 1)) 
    acc_dummy, preds_dummy = evaluate_dummy(dummy_features, test_labels, strategy="most_frequent")
    
    # ---------------------------------------------------------
    # BASELINE 2: Dummy Classifier (Stratified - Random proporcional)
    # ---------------------------------------------------------
    acc_random, preds_random = evaluate_dummy(dummy_features, test_labels, strategy="stratified")
    
    # ---------------------------------------------------------
    # MODELO: resnet50 Entrenado
    # ---------------------------------------------------------
    # Cargar arquitectura
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        acc_model, preds_model, true_labels = evaluate_model(model, test_loader)
        
        # Reporte detallado del modelo
        print("\nReporte de Clasificación (Modelo):")
        print(classification_report(true_labels, preds_model, target_names=CLASS_NAMES, zero_division=0))
        
        # Matriz de Confusión
        cm = confusion_matrix(true_labels, preds_model)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix - Baseline Test')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("baseline_confusion_matrix.png")
        print("Matriz de confusión guardada en 'baseline_confusion_matrix.png'")
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el modelo en {MODEL_PATH}")
        acc_model = 0
    
    # ---------------------------------------------------------
    # COMPARATIVA FINAL
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("RESUMEN DE RESULTADOS (Accuracy)")
    print("="*40)
    print(f"1. Azar Proporcional (Baseline): {acc_random*100:.2f}%")
    print(f"2. Clase Mayoritaria (Baseline): {acc_dummy*100:.2f}%")
    print(f"3. Tu Modelo resnet50:           {acc_model*100:.2f}%")
    print("="*40)
    
    if acc_model > acc_dummy:
        improvement = acc_model - acc_dummy
        print(f"✅ Tu modelo mejora la baseline simple en un {improvement*100:.2f}%")
    else:
        print("⚠️ Tu modelo no supera a la baseline simple. Revisa el entrenamiento.")

if __name__ == "__main__":
    main()
