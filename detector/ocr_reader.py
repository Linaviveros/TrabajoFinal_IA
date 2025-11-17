import cv2
import easyocr

_reader = easyocr.Reader(['en'])  # puedes usar ['en','es'] si quieres

def ocr_placa(cropped_bgr):
    if cropped_bgr is None or cropped_bgr.size == 0:
        return None

    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # detail=0 → solo devuelve el texto, no las cajas
    results = _reader.readtext(rgb, detail=0)
    if not results:
        return None

    text = "".join(results)
    # Solo letras y números, en MAYÚSCULAS
    plate = "".join(ch for ch in text if ch.isalnum()).upper()

    return plate if plate else None
