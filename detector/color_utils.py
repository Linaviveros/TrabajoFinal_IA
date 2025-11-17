# Pico_placa_ia/detector/color_utils.py
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# ========== RUTA AL MODELO (.h5) ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../Pico_placa_ia
MODEL_PATH = PROJECT_ROOT / "Models" / "car_color_model.h5"

# Orden de clases del dataset (no cambiar)
DATASET_CLASSES_EN = [
    "Black",
    "Blue",
    "Brown",
    "Cyan",
    "Green",
    "Grey",
    "Orange",
    "Red",
    "Violet",
    "White",
    "Yellow",
]

# Traducción a español
CLASS_MAP_ES = {
    "Black": "negro",
    "White": "blanco",
    "Grey": "gris",
    "Yellow": "amarillo",
    "Red": "rojo",
    "Blue": "azul",
    "Green": "verde",
    "Brown": "cafe",
    "Orange": "naranja",
    "Cyan": "cian",
    "Violet": "violeta",
}

IMG_SIZE = (64, 64)

# Intentar cargar modelo CNN
try:
    _color_model = load_model(MODEL_PATH)
    print(f"[color_utils] Modelo de color cargado desde {MODEL_PATH}")
except Exception as e:
    _color_model = None
    print(f"[color_utils] NO se pudo cargar el modelo de color: {e}")
    print("[color_utils] Usando método HSV como respaldo.")


# ========== DETECCIÓN POR MODELO CNN ==========
def _detectar_color_por_modelo(img):
    if _color_model is None:
        return None
    if img is None or img.size == 0:
        return None

    try:
        img_resized = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        pred = _color_model.predict(img_input, verbose=0)[0]
        idx = int(np.argmax(pred))

        if idx < 0 or idx >= len(DATASET_CLASSES_EN):
            return None

        cls_en = DATASET_CLASSES_EN[idx]
        return CLASS_MAP_ES.get(cls_en, "indefinido")
    except Exception:
        return None


# ========== DETECCIÓN HSV MEJORADA ==========
def _detectar_color_hsv(img):
    if img is None or img.size == 0:
        return "desconocido"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rangos mejor calibrados para autos
    ranges = {
        "amarillo": [(18, 80, 80), (35, 255, 255)],
        "azul": [(90, 70, 50), (130, 255, 255)],
        "rojo1": [(0, 120, 70), (10, 255, 255)],
        "rojo2": [(170, 120, 70), (180, 255, 255)],
        "blanco": [(0, 0, 200), (180, 40, 255)],
        "negro": [(0, 0, 0), (180, 255, 55)],
        "gris": [(0, 0, 60), (180, 60, 200)],
    }

    max_color = "indefinido"
    max_area = 0

    for color, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        area = cv2.countNonZero(mask)

        if area > max_area:
            max_area = area
            max_color = color

    # Unificar rojo1/rojo2
    if max_color in ["rojo1", "rojo2"]:
        return "rojo"

    return max_color


# ========== API PÚBLICA ==========
def detectar_color(img):
    """
    Intenta CNN; si falla, usa HSV.
    Devuelve color en español como string.
    """
    color_modelo = _detectar_color_por_modelo(img)
    if color_modelo is not None:
        return color_modelo

    return _detectar_color_hsv(img)


def get_vehicle_color(img):
    """
    Wrapper usado en otros módulos. No cambia.
    """
    return detectar_color(img)