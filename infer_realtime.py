"""
Detección de defectos en PCB en tiempo real usando cámara web
Uso: python infer_realtime.py [--camera 0] [--model modelo.pth]
"""

import argparse
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

# Configuración
NUM_CLASSES = 5
MODEL_PATH = "pcb_resnet18_multiclass.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Correcto",
    "Agujero_faltante",
    "Circuito_abierto",
    "Corto_circuito",
    "Rama_cobre",
]

# Colores para cada clase (BGR para OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # ok - verde
    1: (0, 0, 255),      # Missing_hole - rojo
    2: (0, 255, 255),    # Open_circuit - amarillo
    3: (255, 0, 0),      # Short - azul
    4: (255, 0, 255)     # Spur - magenta
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Carga el modelo entrenado"""
    print(f"Cargando modelo desde {model_path}...")
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("✓ Modelo cargado correctamente\n")
    return model

def predict_frame(frame, model):
    """
    Predice la clase de un frame de video
    
    Args:
        frame: Frame BGR de OpenCV
        model: Modelo cargado
    
    Returns:
        pred_class: Índice de la clase predicha
        probabilities: Array con probabilidades
    """
    # Convertir BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Preprocesar
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # Predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probabilities)
    
    return pred_class, probabilities

def draw_predictions(frame, pred_class, probabilities, fps=0):
    """
    Dibuja las predicciones sobre el frame
    
    Args:
        frame: Frame original
        pred_class: Clase predicha
        probabilities: Probabilidades de cada clase
        fps: Frames por segundo
    
    Returns:
        frame con anotaciones
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Color según la clase
    color = CLASS_COLORS[pred_class]
    class_name = CLASS_NAMES[pred_class]
    confidence = probabilities[pred_class] * 100
    
    # Borde de color según la predicción
    thickness = 8 if class_name != "ok" else 4
    cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)
    
    # Panel superior con información principal
    panel_height = 80
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    
    # Texto principal
    label = f"{class_name.upper()}"
    conf_text = f"{confidence:.1f}%"
    
    cv2.putText(overlay, label, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(overlay, conf_text, (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    if fps > 0:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Panel lateral con todas las probabilidades
    panel_width = 280
    x_offset = w - panel_width
    y_start = panel_height + 10
    
    # Fondo semi-transparente para el panel
    cv2.rectangle(overlay, (x_offset, y_start), (w, y_start + 280), (0, 0, 0), -1)
    
    # Título del panel
    cv2.putText(overlay, "PROBABILIDADES:", (x_offset + 5, y_start + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Barras de probabilidad
    bar_y = y_start + 40
    bar_height = 25
    bar_max_width = panel_width - 20
    
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        y = bar_y + i * (bar_height + 5)
        
        # Barra de fondo
        cv2.rectangle(overlay, (x_offset + 5, y), 
                     (x_offset + bar_max_width, y + bar_height), 
                     (50, 50, 50), -1)
        
        # Barra de probabilidad
        bar_width = int(bar_max_width * prob)
        color_bar = CLASS_COLORS[i] if i == pred_class else (100, 100, 100)
        cv2.rectangle(overlay, (x_offset + 5, y), 
                     (x_offset + 5 + bar_width, y + bar_height), 
                     color_bar, -1)
        
        # Texto
        text = f"{name[:12]}: {prob*100:.1f}%"
        cv2.putText(overlay, text, (x_offset + 10, y + 17), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Blending para semi-transparencia
    alpha = 0.85
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Instrucciones en la parte inferior
    instructions = "Presiona 'q' para salir | 's' para captura | 'c' para cambiar camara"
    cv2.putText(frame, instructions, (10, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def list_cameras(max_cameras=10):
    """Lista las cámaras disponibles"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def main():
    parser = argparse.ArgumentParser(description='Detección de defectos PCB en tiempo real')
    parser.add_argument('--camera', type=int, default=0, help='Índice de la cámara (default: 0)')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Ruta al modelo')
    parser.add_argument('--list', action='store_true', help='Listar cámaras disponibles')
    parser.add_argument('--width', type=int, default=1280, help='Ancho del video')
    parser.add_argument('--height', type=int, default=720, help='Alto del video')
    
    args = parser.parse_args()
    
    # Listar cámaras
    if args.list:
        print("Buscando cámaras disponibles...")
        cameras = list_cameras()
        if cameras:
            print(f"Cámaras encontradas: {cameras}")
        else:
            print("No se encontraron cámaras disponibles.")
        return
    
    # Cargar modelo
    model = load_model(args.model)
    
    # Inicializar cámara
    camera_id = args.camera
    print(f"Intentando abrir cámara {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la cámara {camera_id}")
        cameras = list_cameras()
        if cameras:
            print(f"Cámaras disponibles: {cameras}")
            print(f"Usa: python {__file__} --camera {cameras[0]}")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Cámara {camera_id} abierta correctamente")
    print(f"Resolución: {actual_width}x{actual_height}")
    print(f"Dispositivo: {DEVICE}")
    print("\nControles:")
    print("  [q] - Salir")
    print("  [s] - Guardar captura")
    print("  [c] - Cambiar cámara")
    print("  [SPACE] - Pausar/Reanudar")
    print("\n¡Presiona cualquier tecla para comenzar!")
    
    cv2.namedWindow('PCB Defect Detection - Real Time', cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    
    # Variables de control
    frame_count = 0
    start_time = time.time()
    fps = 0
    paused = False
    capture_count = 0
    
    print("\n🎥 Iniciando detección en tiempo real...\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error al leer el frame")
                break
            
            # Calcular FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                start_time = time.time()
            
            # Realizar predicción
            pred_class, probabilities = predict_frame(frame, model)
            
            # Dibujar resultados
            annotated_frame = draw_predictions(frame, pred_class, probabilities, fps)
            current_frame = annotated_frame
        
        # Mostrar frame
        cv2.imshow('PCB Defect Detection - Real Time', current_frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Saliendo...")
            break
        
        elif key == ord('s'):
            # Guardar captura
            capture_count += 1
            filename = f"capture_{capture_count}_{CLASS_NAMES[pred_class]}.png"
            cv2.imwrite(filename, current_frame)
            print(f"📸 Captura guardada: {filename}")
        
        elif key == ord('c'):
            # Cambiar cámara
            print("\nCámaras disponibles:", list_cameras())
            try:
                new_camera = int(input("Ingrese número de cámara: "))
                cap.release()
                cap = cv2.VideoCapture(new_camera)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                    camera_id = new_camera
                    print(f"✓ Cambiado a cámara {camera_id}")
                else:
                    print(f"❌ No se pudo abrir cámara {new_camera}, volviendo a {camera_id}")
                    cap = cv2.VideoCapture(camera_id)
            except (ValueError, KeyboardInterrupt):
                print("Operación cancelada")
        
        elif key == ord(' '):
            # Pausar/Reanudar
            paused = not paused
            status = "PAUSADO" if paused else "REANUDADO"
            print(f"⏸️  {status}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Recursos liberados correctamente")

if __name__ == "__main__":
    main()
