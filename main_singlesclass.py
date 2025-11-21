
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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchviz import make_dot

# -------- CONFIG ----------
DATA_DIR = "pcb-defects"   # carpeta donde descomprimiste el dataset
MY_IMAGE = "image.png"     # tu imagen de muestra
MY_IMAGE2 = "image2.png"     # tu imagen de muestra
NUM_CLASSES = 2            # defective / ok
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL = "pcb_resnet18.pth"
# --------------------------

def build_image_list(data_dir):
    img_paths = []
    labels = []
    # Intenta recorrer carpetas; si no, toma todos png/jpg y decide label por nombre
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = Path(root) / f
                name = f.lower()
                # heurística: si el nombre contiene 'ok' o 'good' -> class 0 (no defect)
                # si contiene 'defect' o alguno de los tipos -> class 1 (defect)
                if "PCB_USED" in root:
                    lbl = 0
                    for i in range(2):         
                        img_paths.append(str(p))
                        labels.append(lbl)
                else:
                    # Ignorar Mouse_bite
                    if "Mouse_bite" in root:
                        continue
                        
                    # si dataset usa tipos: missing_hole, open, short, spur, spurious_copper
                    lbl = 1
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
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

img_paths, labels = build_image_list(DATA_DIR)
print(f"Found {len(img_paths)} images.")

# If too few images of one class, warn
n_def = sum(labels)
n_ok = len(labels) - n_def
print("Defective:", n_def, "OK:", n_ok)

# Create dataset & splits
# Estrategia de división estratificada
indices = np.arange(len(labels))
train_idx, temp_idx = train_test_split(
    indices, 
    test_size=0.3, 
    stratify=labels, 
    random_state=42
)

temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.5, 
    stratify=temp_labels, 
    random_state=42
)

def get_subset(paths, labs, idxs):
    return [paths[i] for i in idxs], [labs[i] for i in idxs]

train_paths, train_labels = get_subset(img_paths, labels, train_idx)
val_paths, val_labels = get_subset(img_paths, labels, val_idx)
test_paths, test_labels = get_subset(img_paths, labels, test_idx)

# Crear datasets independientes
train_set = PCBClassDataset(train_paths, train_labels, transform=train_tf)
val_set = PCBClassDataset(val_paths, val_labels, transform=val_tf)
test_set = PCBClassDataset(test_paths, test_labels, transform=val_tf)

print(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Model (resnet18 fine-tune) ---
model = models.resnet18(pretrained=True)
# replace final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
best_val = 0.0
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
    # validation
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
    print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f} val_acc={acc:.4f}")
    if acc > best_val:
        best_val = acc
        torch.save(model.state_dict(), OUT_MODEL)
        print("Saved best model.")

# --- Evaluación en test y matriz de confusión ---
# cargar mejor modelo
# --- Evaluación en test y matriz de confusión ---
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
        # Para binario, nos interesa la probabilidad de la clase positiva (1)
        y_scores.extend(probs.cpu().numpy()[:, 1].tolist())

# ⚡️ Fuerza a sklearn a usar ambas clases incluso si falta una
labels_all = [0, 1]
cm = confusion_matrix(y_true, y_pred, labels=labels_all)
print("Confusion Matrix:\n", cm)

print(classification_report(y_true, y_pred, labels=labels_all, target_names=["ok", "defective"]))

# --- ROC & AUC ---
print("\nGenerando curva ROC...")
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Binary)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("roc_curve_binary.png")
print("Saved roc_curve_binary.png")


# --- Guardar matriz como figura ---
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks([0,1], ["ok","defective"])
plt.yticks([0,1], ["ok","defective"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# --- Inferencia rápida sobre tu imagen (image.png) ---
def predict_image(path, model, transform):
    model.eval()
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_t)
        pred = torch.argmax(out, dim=1).item()
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return pred, probs

if os.path.exists(MY_IMAGE):
    pred, probs = predict_image(MY_IMAGE, model, val_tf)
    label = "defective" if pred==1 else "ok"
    print(f"Prediction for {MY_IMAGE}: {label} (probs: {probs})")
else:
    print(f"{MY_IMAGE} not found in working dir. Copy it or set MY_IMAGE variable accordingly.")

if os.path.exists(MY_IMAGE2):
    pred, probs = predict_image(MY_IMAGE2, model, val_tf)
    label = "defective" if pred==1 else "ok"
    print(f"Prediction for {MY_IMAGE2}: {label} (probs: {probs})")
else:
    print(f"{MY_IMAGE2} not found in working dir. Copy it or set MY_IMAGE2 variable accordingly.")

# --- Generar GRAFO del modelo (torchviz) ---
print("\nGenerating model graph with torchviz...")

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

model_graph = make_dot(
    model(dummy_input),
    params=dict(model.named_parameters()),
    show_attrs=True,
    show_saved=True
)

model_graph.render("pcb_model_graph", format="png")
print("Saved pcb_model_graph.png")
