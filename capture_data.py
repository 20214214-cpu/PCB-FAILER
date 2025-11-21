import cv2
import os
import time
import argparse
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = "pcb-defects"

# Mapeo de teclas a (Nombre Clase, Ruta Relativa)
# Ajusta las rutas según tu estructura real
CLASSES = {
    '0': {"name": "PCB_USED (OK)",     "path": os.path.join(BASE_DIR, "PCB_USED")},
    '1': {"name": "Missing_hole",      "path": os.path.join(BASE_DIR, "images", "Missing_hole")},
    '2': {"name": "Open_circuit",      "path": os.path.join(BASE_DIR, "images", "Open_circuit")},
    '3': {"name": "Short",             "path": os.path.join(BASE_DIR, "images", "Short")},
    '4': {"name": "Spur",              "path": os.path.join(BASE_DIR, "images", "Spur")},
}

def create_directories():
    """Crea las carpetas si no existen"""
    print("Verificando directorios...")
    for key, info in CLASSES.items():
        os.makedirs(info["path"], exist_ok=True)
        print(f"  [OK] {info['path']}")

def main():
    parser = argparse.ArgumentParser(description="Herramienta de captura de datos PCB")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0)")
    args = parser.parse_args()

    create_directories()
    
    # Intentar abrir la cámara
    # 0 suele ser la webcam integrada, 1 una externa USB
    print(f"\nIniciando cámara {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara (índice 0).")
        print("Intenta cambiar cv2.VideoCapture(0) a cv2.VideoCapture(1) en el código.")
        return

    # Configurar resolución (opcional, intenta HD si está disponible)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== SISTEMA DE CAPTURA DE DATOS PCB ===")
    print("Instrucciones:")
    for key, info in CLASSES.items():
        print(f"  Presiona [{key}] para guardar en: {info['name']}")
    print("  Presiona [q] para SALIR")
    print("=======================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame de la cámara.")
            break

        # Crear una copia para dibujar la interfaz (GUI)
        gui_frame = frame.copy()
        
        # Dibujar instrucciones en pantalla
        y_start = 30
        cv2.putText(gui_frame, "CONTROLES DE GUARDADO:", (10, y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (key, info) in enumerate(CLASSES.items()):
            text = f"[{key}] {info['name']}"
            y_pos = y_start + 25 + (i * 25)
            # Sombra negra para mejor legibilidad
            cv2.putText(gui_frame, text, (11, y_pos+1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Texto color
            color = (0, 255, 0) if "OK" in info['name'] else (0, 255, 255)
            cv2.putText(gui_frame, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(gui_frame, "[q] Salir", (10, y_start + 25 + (len(CLASSES) * 25) + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mostrar el frame
        cv2.imshow('Captura PCB - Presiona tecla para clasificar', gui_frame)

        # Esperar input de teclado (1ms)
        key_code = cv2.waitKey(1) & 0xFF
        
        # Salir
        if key_code == ord('q'):
            break
        
        # Verificar si se presionó una tecla de clase
        # Convertir código de tecla a caracter si es posible
        try:
            key_char = chr(key_code)
        except:
            key_char = None
        
        if key_char in CLASSES:
            info = CLASSES[key_char]
            
            # Generar nombre único
            timestamp = int(time.time() * 1000)
            # Limpiar nombre para archivo
            safe_name = info['name'].split()[0].replace("/", "_")
            filename = f"{safe_name}_{timestamp}.jpg"
            filepath = os.path.join(info["path"], filename)
            
            # Guardar imagen original (sin textos)
            cv2.imwrite(filepath, frame)
            print(f"Guardada imagen en: {filepath}")
            
            # Feedback visual de guardado
            cv2.rectangle(gui_frame, (0, 0), (gui_frame.shape[1], gui_frame.shape[0]), (0, 255, 0), 10)
            cv2.putText(gui_frame, f"GUARDADO: {info['name']}", (50, gui_frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('Captura PCB - Presiona tecla para clasificar', gui_frame)
            cv2.waitKey(300) # Pausa para ver el feedback

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
