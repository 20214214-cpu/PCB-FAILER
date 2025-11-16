import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot

# -------- CONFIG ----------
DATA_DIR = "pcb-defects"
MY_IMAGE = "image.png"
MY_IMAGE2 = "image2.png"
NUM_CLASSES = 7  # 6 tipos de defectos + ok
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL = "pcb_resnet18_multiclass.pth"

# Mapeo de clases
CLASS_NAMES = [
    "ok",                  # 0
    "Missing_hole",        # 1
    "Mouse_bite",          # 2
    "Open_circuit",        # 3
    "Short",               # 4
    "Spur",                # 5
    "Spurious_copper"      # 6
]

DEFECT_FOLDERS = {
    "Missing_hole": 1,
    "Mouse_bite": 2,
    "Open_circuit": 3,
    "Short": 4,
    "Spur": 5,
    "Spurious_copper": 6
}
# --------------------------

def build_image_list(data_dir):
    """
    Construye lista de imágenes con sus etiquetas multiclase
    - PCB_USED -> clase 0 (ok)
    - Cada carpeta de defecto -> clase específica (1-6)
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

# --- Transforms ---
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
dataset = PCBClassDataset(img_paths, labels, transform=None)
n = len(dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

# Attach transforms
train_set.dataset.transform = train_tf
val_set.dataset.transform = val_tf
test_set.dataset.transform = val_tf

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Model (ResNet18 fine-tune) ---
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Calcular pesos para clases desbalanceadas
class_weights = []
total = len(labels)
for i in range(NUM_CLASSES):
    count = sum(1 for lbl in labels if lbl == i)
    weight = total / (NUM_CLASSES * count) if count > 0 else 1.0
    class_weights.append(weight)

class_weights = torch.FloatTensor(class_weights).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
print("\n=== Entrenamiento ===")
best_val = 0.0
train_losses = []
val_accs = []

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
    
    print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f} val_acc={acc:.4f}")
    
    if acc > best_val:
        best_val = acc
        torch.save(model.state_dict(), OUT_MODEL)
        print("✓ Saved best model.")

# --- Gráficas de entrenamiento ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), val_accs, marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
print("\nSaved training_history.png")

# --- Evaluación en test ---
print("\n=== Evaluación en Test ===")
model.load_state_dict(torch.load(OUT_MODEL, map_location=DEVICE))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labs in test_loader:
        imgs = imgs.to(DEVICE)
        outs = model(imgs)
        preds = torch.argmax(outs, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labs.numpy().tolist())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), 
                          target_names=CLASS_NAMES, zero_division=0))

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
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

model_graph = make_dot(
    model(dummy_input),
    params=dict(model.named_parameters()),
    show_attrs=True,
    show_saved=True
)

model_graph.render("pcb_model_graph_multiclass", format="png")
print("Saved pcb_model_graph_multiclass.png")

print("\n✓ Proceso completado.")
