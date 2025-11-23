import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing # Importación necesaria para el soporte de workers/freeze_support

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler # Importación necesaria
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot
from sklearn.preprocessing import label_binarize
from itertools import cycle

# -------- CONFIG ----------
DATA_DIR = "pcb-defects"
MY_IMAGE = "image.png"
MY_IMAGE2 = "image2.png"
NUM_CLASSES = 5  # 4 tipos de defectos + ok
BATCH_SIZE = 16
EPOCHS = 50  # Entrenamiento extendido
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL = "pcb_resnet50_multiclass.pth"
BALANCE_OK_CLASS = False  # Si es True, replica imágenes OK 3x para balanceo
ENHANCE_WEIGHT_OK_CLASS = False  # Si es True, aumenta peso de clase OK en la pérdida
OK_REPLICATION_FACTOR = 1.05  # Factor de replicación para clase OK

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Detener si no mejora en N epochs
LR_REDUCE_PATIENCE = 3  # Reducir LR si no mejora en N epochs
LR_REDUCE_FACTOR = 0.5  # Factor de reducción del LR
MIN_DELTA = 0.0001  # Mejora mínima para considerar progreso

# Mapeo de clases
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
# --------------------------

def build_image_list(data_dir):
    """
    Construye lista de imágenes con sus etiquetas multiclase
    - PCB_USED -> clase 0 (ok) - Opcionalmente replicada según BALANCE_OK_CLASS
    - Cada carpeta de defecto -> clase específica (1-5)
    """
    img_paths = []
    labels = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = Path(root) / f
                
                # Determinar la clase según la carpeta
                if "PCB_USED" in root:
                    lbl = 0  # ok
                    # Replicar imágenes OK si está activado el balanceo
                    replication = int(OK_REPLICATION_FACTOR) if BALANCE_OK_CLASS and OK_REPLICATION_FACTOR > 1 else 1 # Se asegura de que sea un entero para el range
                    for _ in range(replication):
                        img_paths.append(str(p))
                        labels.append(lbl)
                else:
                    # Buscar en qué carpeta de defecto está
                    lbl = 0  # default ok
                    for defect_name, defect_label in DEFECT_FOLDERS.items():
                        if defect_name in root:
                            lbl = defect_label
                            break
                    
                    img_paths.append(str(p))
                    labels.append(lbl)
    
    return img_paths, labels

# --- Dataset PyTorch ---
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

# --- Transforms - Aumentaciones específicas para PCB ---
train_tf = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(5), # Baja la rotación
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.1), # Opcional: Probar si mejora el rendimiento
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), # O más suave
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Menos agresivo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # Soporte para multiprocessing en Windows (Workers de DataLoader)
    multiprocessing.freeze_support() 

    # Construir dataset
    img_paths, labels = build_image_list(DATA_DIR)
    print(f"Found {len(img_paths)} images.")

    # Mostrar distribución de clases
    class_counts = {name: 0 for name in CLASS_NAMES}
    for lbl in labels:
        class_counts[CLASS_NAMES[lbl]] += 1

    print("\n=== Distribución de clases ===")
    for name, count in class_counts.items():
        print(f"{name}: {count}")

    # Create dataset & splits
    # Estrategia de división estratificada para asegurar % de cada clase en test
    # 1. Separar Train (70%) del resto (30%)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.3, 
        stratify=labels, 
        random_state=42
    )

    # 2. Separar Val (15%) y Test (15%) del resto (que es el 30% original -> dividir a la mitad)
    # Necesitamos las etiquetas correspondientes a temp_idx para estratificar
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        stratify=temp_labels, 
        random_state=42
    )

    # Helper para filtrar listas
    def get_subset(paths, labs, idxs):
        return [paths[i] for i in idxs], [labs[i] for i in idxs]

    train_paths, train_labels = get_subset(img_paths, labels, train_idx)
    val_paths, val_labels = get_subset(img_paths, labels, val_idx)
    test_paths, test_labels = get_subset(img_paths, labels, test_idx)

    # Crear datasets independientes con sus transformaciones correctas
    train_set = PCBClassDataset(train_paths, train_labels, transform=train_tf)
    val_set = PCBClassDataset(val_paths, val_labels, transform=val_tf)
    test_set = PCBClassDataset(test_paths, test_labels, transform=val_tf)

    print(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Calcular pesos de muestreo para balancear clases en el entrenamiento
    train_class_counts = {name: 0 for name in CLASS_NAMES}
    for lbl in train_labels:
        train_class_counts[CLASS_NAMES[lbl]] += 1
    
    # Convertir counts a lista ordenada por índice de clase (0 a NUM_CLASSES-1)
    ordered_counts = [train_class_counts[CLASS_NAMES[i]] for i in range(NUM_CLASSES)]
    
    # Calcular pesos inversos (ejemplo: si la clase 0 tiene 1000 y la 1 tiene 100, la 1 debe pesar 10 veces más)
    total_samples = len(train_labels)
    class_weights = [total_samples / (c * NUM_CLASSES) if c > 0 else 1.0 for c in ordered_counts]
    class_weights_map = {i: class_weights[i] for i in range(NUM_CLASSES)}

    # Asignar el peso a cada muestra en el conjunto de entrenamiento
    sample_weights = [class_weights_map[lbl] for lbl in train_labels]
    
    # Crear el Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    print(f"\nClases en Train: {train_class_counts}")
    print("Usando WeightedRandomSampler para balancear los batches automáticamente.")

    # IMPORTANTE: shuffle=False cuando se usa sampler
    # num_workers > 0 solo funciona con if __name__ == "__main__" en Windows.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2) 
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Para Test se puede dejar en 0

    # --- Model (resnet50 fine-tune) ---
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Calcular pesos para la función de pérdida (opcionalmente)
    # Se usa la distribución de las etiquetas originales (antes del split)
    # Si se usó WeightedRandomSampler, este peso no es estrictamente necesario, 
    # pero ayuda a penalizar errores en clases subrepresentadas.
    loss_class_counts = [class_counts[CLASS_NAMES[i]] for i in range(NUM_CLASSES)]
    total = len(labels)
    
    loss_class_weights = []
    for count in loss_class_counts:
        # Peso inverso de la frecuencia
        weight = total / (NUM_CLASSES * count) if count > 0 else 1.0
        loss_class_weights.append(weight)

    if ENHANCE_WEIGHT_OK_CLASS:
        ok_class_index = CLASS_NAMES.index("ok")
        loss_class_weights[ok_class_index] *= OK_REPLICATION_FACTOR

    loss_class_weights = torch.FloatTensor(loss_class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training loop con Early Stopping ---
    print("\n=== Entrenamiento ===")
    print(f"Early Stopping: Patience={EARLY_STOPPING_PATIENCE}, Min Delta={MIN_DELTA}")

    best_val = 0.0
    train_losses = []
    val_accs = []
    epochs_no_improve = 0
    lr_epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - train"):
            imgs = imgs.to(DEVICE)
            labs = labs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
        
        # len(train_loader.dataset) es más robusto si el dataset es replicado
        epoch_loss = running_loss / len(train_loader.dataset) 
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(DEVICE)
                labs = labs.to(DEVICE)
                outs = model(imgs)
                preds = torch.argmax(outs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labs.cpu().numpy())
        
        acc = accuracy_score(val_trues, val_preds)
        val_accs.append(acc)
        
        print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f} val_acc={acc:.4f}", end="")
        
        # Early Stopping Logic
        if acc > best_val + MIN_DELTA:
            best_val = acc
            epochs_no_improve = 0
            lr_epochs_no_improve = 0
            torch.save(model.state_dict(), OUT_MODEL)
            print(" ✓ Saved best model.")
        else:
            epochs_no_improve += 1
            lr_epochs_no_improve += 1
            print(f" (no improvement: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})", end="")

            if lr_epochs_no_improve >= LR_REDUCE_PATIENCE:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= LR_REDUCE_FACTOR
                current_lr = optimizer.param_groups[0]['lr']
                print(f" → LR reducido a {current_lr:.2e}", end="")
                lr_epochs_no_improve = 0

            print()
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val:.4f}")
                break

    # --- Gráficas de entrenamiento ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accs)+1), val_accs, marker='o', color='orange')
    plt.axhline(y=best_val, color='r', linestyle='--', label=f'Best: {best_val:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("\nSaved training_history.png")

    # --- Evaluación en test ---
    print("\n=== Evaluación en Test ===")
    model.load_state_dict(torch.load(OUT_MODEL, map_location=DEVICE))
    model.eval()
    y_true, y_pred = [], []
    y_scores = []

    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(DEVICE)
            outs = model(imgs)
            probs = torch.softmax(outs, dim=1)
            preds = torch.argmax(outs, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labs.numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), 
                              target_names=CLASS_NAMES, zero_division=0))

    # --- ROC & AUC ---
    print("\nGenerando curvas ROC...")
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    y_scores = np.array(y_scores)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve_multiclass.png")
    print("Saved roc_curve_multiclass.png")

    # --- Visualizar matriz de confusión ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Multiclass PCB Defect Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix_multiclass.png")
    print("Saved confusion_matrix_multiclass.png")

    # --- Inferencia sobre imágenes específicas ---
    def predict_image(path, model, transform):
        """Predice la clase de una imagen"""
        model.eval()
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = model(img_t)
            pred = torch.argmax(out, dim=1).item()
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        
        return pred, probs

    print("\n=== Predicciones en imágenes de prueba ===")
    for img_path in [MY_IMAGE, MY_IMAGE2]:
        if os.path.exists(img_path):
            pred, probs = predict_image(img_path, model, val_tf)
            label = CLASS_NAMES[pred]
            confidence = probs[pred] * 100
            
            print(f"\n{img_path}:")
            print(f"  Predicción: {label} ({confidence:.2f}% confianza)")
            print("  Probabilidades por clase:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
                print(f"    {class_name}: {prob*100:.2f}%")
        else:
            print(f"\n{img_path} no encontrado.")

    # --- Generar grafo del modelo ---
    print("\n=== Generando grafo del modelo ===")
    dummy_input = torch.randn(1, 3, 400,400).to(DEVICE)

    model_graph = make_dot(
        model(dummy_input),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )

    model_graph.render("pcb_model_graph_multiclass", format="png")
    print("Saved pcb_model_graph_multiclass.png")

    print("\n✓ Proceso completado.")