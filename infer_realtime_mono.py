"""
Detección de defectos en PCB (binaria) en tiempo real usando cámara web
Uso:
    python infer_realtime_monoclass.py --camera 0 --model pcb_resnet18.pth
"""

import argparse
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

# ================== CONFIG ==================
NUM_CLASSES = 2
MODEL_PATH = "pcb_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0 = OK, 1 = Defectuoso
CLASS_NAMES = [
    "Correcto",
    "Defectuoso",
]

# Colores para cada clase (BGR para OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),   # Correcto - verde
    1: (0, 0, 255),   # Defectuoso - rojo
}

# Transform (igual que en entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ============================================

def load_model(model_path):
    """Carga el modelo entrenado binario (2 clases)."""
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
    Predice la clase de un frame de video (0 = Correcto, 1 = Defectuoso).
    """
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Preprocesar
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probabilities))

    return pred_class, probabilities

def draw_predictions(frame, pred_class, probabilities, fps=0.0):
    """
    Dibuja las predicciones sobre el frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Info principal
    color = CLASS_COLORS[pred_class]
    class_name = CLASS_NAMES[pred_class]
    confidence = probabilities[pred_class] * 100

    # Borde de color según la predicción
    thickness = 4 if pred_class == 0 else 8  # más grueso si es defectuoso
    cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)

    # Panel superior
    panel_height = 80
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)

    # Texto principal
    label = class_name.upper()
    conf_text = f"{confidence:.1f}%"

    cv2.putText(overlay, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(overlay, conf_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # FPS
    if fps > 0:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel lateral de probabilidades (solo 2 barras)
    panel_width = 260
    x_offset = w - panel_width
    y_start = panel_height + 10
    panel_height_prob = 90

    cv2.rectangle(overlay, (x_offset, y_start),
                  (w, y_start + panel_height_prob), (0, 0, 0), -1)

    cv2.putText(overlay, "PROBABILIDADES:", (x_offset + 5, y_start + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    bar_y = y_start + 35
    bar_height = 20
    bar_max_width = panel_width - 20

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        y = bar_y + i * (bar_height + 8)

        # fondo
        cv2.rectangle(overlay, (x_offset + 5, y),
                      (x_offset + bar_max_width, y + bar_height),
                      (50, 50, 50), -1)

        # barra
        bar_width = int(bar_max_width * prob)
        color_bar = CLASS_COLORS[i] if i == pred_class else (100, 100, 100)
        cv2.rectangle(overlay, (x_offset + 5, y),
                      (x_offset + 5 + bar_width, y + bar_height),
                      color_bar, -1)

        text = f"{name}: {prob*100:.1f}%"
        cv2.putText(overlay, text, (x_offset + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Blending
    alpha = 0.85
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Instrucciones
    instructions = "Presiona 'q' para salir | 's' captura | 'c' cambiar camara | SPACE pausa"
    cv2.putText(frame, instructions, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def list_cameras(max_cameras=10):
    """Lista las cámaras disponibles."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def main():
    parser = argparse.ArgumentParser(
        description='Detección binaria de defectos PCB en tiempo real'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Índice de la cámara (default: 0)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Ruta al modelo (.pth)')
    parser.add_argument('--list', action='store_true',
                        help='Listar cámaras disponibles')
    parser.add_argument('--width', type=int, default=1280,
                        help='Ancho del video')
    parser.add_argument('--height', type=int, default=720,
                        help='Alto del video')

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
            print(f"Usa: python infer_realtime_monoclass.py --camera {cameras[0]}")
        return

    # Config resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Cámara {camera_id} abierta correctamente")
    print(f"Resolución: {actual_width}x{actual_height}")
    print(f"Dispositivo: {DEVICE}")
    print("\nControles:")
    print("  [q]     - Salir")
    print("  [s]     - Guardar captura")
    print("  [c]     - Cambiar cámara")
    print("  [SPACE] - Pausar/Reanudar")
    print("\n¡Presiona cualquier tecla en la ventana para comenzar!")

    cv2.namedWindow('PCB Binary Defect Detection - Real Time', cv2.WINDOW_NORMAL)
    cv2.waitKey(0)

    # Variables de control
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    paused = False
    capture_count = 0

    print("\n🎥 Iniciando detección en tiempo real...\n")

    current_frame = None
    pred_class = 0
    probabilities = np.array([1.0, 0.0])

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("❌ Error al leer el frame")
                break

            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30.0 / elapsed if elapsed > 0 else 0.0
                start_time = time.time()

            # Predicción
            pred_class, probabilities = predict_frame(frame, model)

            # Dibujar resultados
            current_frame = draw_predictions(frame, pred_class, probabilities, fps)

        # Mostrar frame
        cv2.imshow('PCB Binary Defect Detection - Real Time', current_frame)

        # Teclas
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 Saliendo...")
            break

        elif key == ord('s'):
            capture_count += 1
            filename = f"capture_{capture_count}_{CLASS_NAMES[pred_class]}.png"
            cv2.imwrite(filename, current_frame)
            print(f"📸 Captura guardada: {filename}")

        elif key == ord('c'):
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
            paused = not paused
            status = "PAUSADO" if paused else "REANUDADO"
            print(f"⏸️  {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Recursos liberados correctamente")

if __name__ == "__main__":
    main()
