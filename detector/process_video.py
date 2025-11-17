# detector/live_camera.py

import os
import cv2
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from detector.ocr_reader import ocr_placa
from detector.color_utils import get_vehicle_color
from Reglas.pico_placa_pasto_2026 import puede_circular_pasto
from db.db import init_db, upsert_vehicle, insert_detection, insert_violation

BASE_DIR = Path(__file__).resolve().parent.parent
FOTOMULTAS_DIR = BASE_DIR / "outputs" / "fotomultas"
FOTOMULTAS_DIR.mkdir(parents=True, exist_ok=True)

# Clases de COCO que consideramos vehículos
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}

# Índice de cámara que quieres usar (0, 1, 2...)
CAM_INDEX = 1  # cámbialo si tu webcam real es otra


def live_camera(
    plate_model_path: str | None = None,
    vehicle_model_path: str | None = None,
):
    """
    Solo cámara en vivo:

    - Detecta vehículos.
    - Dentro de cada vehículo busca placa.
    - OCR de la placa.
    - Color del vehículo.
    - Reglas de pico y placa Pasto.
    - Guarda detecciones y multas en la BD.
    """

    init_db()

    # Modelo de PLACAS: aquí pones el que mejor te funcione.
    if plate_model_path is None:
        plate_model_path = str(BASE_DIR / "Models" / "yolov8_placas_best.pt")
        # Si ves que el viejo funciona mejor, cambia a:
        # plate_model_path = str(BASE_DIR / "Models" / "license_plate_detector.pt")

    # Modelo de VEHÍCULOS (COCO, por ejemplo)
    if vehicle_model_path is None:
        vehicle_model_path = str(BASE_DIR / "Models" / "yolov8s.pt")

    print("📷 Iniciando cámara en vivo...")
    print("  ▶ Modelo placas:", plate_model_path)
    print("  ▶ Modelo vehículos:", vehicle_model_path)

    plate_model = YOLO(plate_model_path)
    veh_model = YOLO(vehicle_model_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {CAM_INDEX}")

    cooldown = {}
    COOLDOWN_SEC = 10
    video_name = "<live_camera>"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ No se pudo leer frame de la cámara")
            break

        ts = datetime.now()
        h, w = frame.shape[:2]

        # 1) Detectar vehículos en el frame completo
        vr = veh_model(frame, conf=0.35, verbose=False)[0]
        vehicles = []
        if vr.boxes is not None:
            names = vr.names
            for b in vr.boxes:
                cls_id = int(b.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    vehicles.append((x1, y1, x2, y2, cls_name))

        # Debug rápido
        # print(f"[DEBUG] Vehículos detectados: {len(vehicles)}")

        for (vx1, vy1, vx2, vy2, vname) in vehicles:
            # Dibujar vehículo (azul)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                vname,
                (vx1, max(20, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue

            # 2) Buscar placa SOLO dentro del vehículo
            pr = plate_model(vehicle_crop, conf=0.25, verbose=False)[0]
            if pr.boxes is None or len(pr.boxes) == 0:
                continue

            for pbox in pr.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])

                # Coordenadas globales
                gx1, gy1 = vx1 + px1, vy1 + py1
                gx2, gy2 = vx1 + px2, vy1 + py2

                gx1 = max(0, min(gx1, w - 1))
                gx2 = max(0, min(gx2, w - 1))
                gy1 = max(0, min(gy1, h - 1))
                gy2 = max(0, min(gy2, h - 1))
                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                plate_crop = frame[gy1:gy2, gx1:gx2]
                if plate_crop.size == 0:
                    continue

                # Dibujar SIEMPRE la caja de placa (amarillo por defecto)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                label = "PLACA?"

                # 3) OCR de la placa
                plate_text = ocr_placa(plate_crop)
                if plate_text:
                    # 4) Color sobre el vehículo completo
                    color = get_vehicle_color(vehicle_crop)

                    # 5) Reglas pico y placa Pasto
                    ok, detalle = puede_circular_pasto(plate_text, ts)
                    violation = not ok
                    motivo = ""
                    if isinstance(detalle, dict):
                        motivo = detalle.get("motivo", "")

                    # 6) Guardar en BD
                    upsert_vehicle(plate_text, color, ts)

                    bbox_str = f"{gx1},{gy1},{gx2},{gy2}"
                    photo_path = None

                    if violation:
                        last_t = cooldown.get(plate_text)
                        if (not last_t) or ((ts - last_t).total_seconds() > COOLDOWN_SEC):
                            cooldown[plate_text] = ts
                            photo_path = (
                                FOTOMULTAS_DIR
                                / f"{plate_text}_{ts:%Y%m%d_%H%M%S}.jpg"
                            )
                            cv2.imwrite(str(photo_path))
                            insert_violation(
                                plate_text, color, ts, motivo, str(photo_path)
                            )

                    insert_detection(
                        plate_text,
                        color,
                        ts,
                        video_name,
                        bbox_str,
                        violation,
                        motivo,
                        str(photo_path) if photo_path else None,
                    )

                    # Cambiar color de la placa según infracción
                    if violation:
                        cv2.rectangle(
                            frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2
                        )
                        label = f"{plate_text} ({color}) EN FALTA"
                    else:
                        cv2.rectangle(
                            frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2
                        )
                        label = f"{plate_text} ({color})"

                # Mostrar texto (aunque el OCR falle)
                cv2.putText(
                    frame,
                    label,
                    (gx1, max(20, gy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Pico y Placa IA - Camara en vivo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cámara detenida. Datos guardados en la base de datos.")


if __name__ == "__main__":
    live_camera()
