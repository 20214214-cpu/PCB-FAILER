"""
Script de inferencia para clasificación multiclase de defectos en PCB
Uso: python infer_multiclass.py --image ruta/a/imagen.png
"""

import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Configuración
NUM_CLASSES = 5
MODEL_PATH = "pcb_resnet50_multiclass.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "ok",
    "Missing_hole",
    "Open_circuit",
    "Short",
    "Spur"
]

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Carga el modelo entrenado"""
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(image_path, model, show_plot=True):
    """
    Predice la clase de defecto en una imagen
    
    Args:
        image_path: Ruta a la imagen
        model: Modelo cargado
        show_plot: Si True, muestra visualización
    
    Returns:
        pred_class: Índice de la clase predicha
        class_name: Nombre de la clase
        probabilities: Array con probabilidades de cada clase
    """
    # Cargar y preprocesar imagen
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probabilities)
    
    class_name = CLASS_NAMES[pred_class]
    
    # Visualización
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Imagen original
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'Predicción: {class_name}\nConfianza: {probabilities[pred_class]*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        
        # Gráfico de barras con probabilidades
        colors = ['green' if i == pred_class else 'skyblue' for i in range(NUM_CLASSES)]
        bars = ax2.barh(CLASS_NAMES, probabilities * 100, color=colors)
        ax2.set_xlabel('Probabilidad (%)', fontsize=12)
        ax2.set_title('Distribución de Probabilidades', fontsize=14)
        ax2.set_xlim(0, 100)
        
        # Añadir valores en las barras
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{prob*100:.1f}%', 
                    ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'prediction_{pred_class}_{class_name}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualización guardada como: prediction_{pred_class}_{class_name}.png")
        plt.show()
    
    return pred_class, class_name, probabilities

def batch_predict(image_paths, model):
    """Predice múltiples imágenes"""
    results = []
    
    for img_path in image_paths:
        try:
            pred_class, class_name, probs = predict_image(img_path, model, show_plot=False)
            results.append({
                'image': img_path,
                'predicted_class': class_name,
                'confidence': probs[pred_class] * 100,
                'probabilities': probs
            })
            print(f"{img_path}: {class_name} ({probs[pred_class]*100:.2f}%)")
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Clasificador de defectos en PCB')
    parser.add_argument('--image', type=str, help='Ruta a imagen individual')
    parser.add_argument('--batch', type=str, nargs='+', help='Lista de imágenes')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Ruta al modelo')
    parser.add_argument('--no-plot', action='store_true', help='No mostrar visualización')
    
    args = parser.parse_args()
    
    # Cargar modelo
    print(f"Cargando modelo desde {args.model}...")
    model = load_model(args.model)
    print("✓ Modelo cargado correctamente\n")
    
    # Predicción individual
    if args.image:
        print(f"Analizando imagen: {args.image}")
        pred_class, class_name, probs = predict_image(args.image, model, show_plot=not args.no_plot)
        
        print(f"\n{'='*50}")
        print(f"RESULTADO:")
        print(f"  Clase predicha: {class_name}")
        print(f"  Confianza: {probs[pred_class]*100:.2f}%")
        print(f"\nProbabilidades completas:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            print(f"  {name:20s}: {prob*100:6.2f}%")
        print(f"{'='*50}\n")
    
    # Predicción por lotes
    elif args.batch:
        print(f"Analizando {len(args.batch)} imágenes en lote...\n")
        results = batch_predict(args.batch, model)
        
        print(f"\n{'='*50}")
        print("RESUMEN DE RESULTADOS:")
        for r in results:
            print(f"  {r['image']}: {r['predicted_class']} ({r['confidence']:.2f}%)")
        print(f"{'='*50}\n")
    
    else:
        print("Error: Especifica --image o --batch")
        parser.print_help()

if __name__ == "__main__":
    main()
