# detector/live_camera.py

import cv2
from datetime import datetime, timedelta
from pathlib import Path

from ultralytics import YOLO
from detector.ocr_reader import ocr_placa
from detector.color_utils import get_vehicle_color
from Reglas.pico_placa_pasto_2026 import puede_circular_pasto
from db.db import (
    init_db,
    upsert_vehicle,
    insert_detection,
    insert_violation,
    get_plate_status,   # 👈 para saber si ya estaba en la BD
)

BASE_DIR = Path(__file__).resolve().parent.parent
FOTOMULTAS_DIR = BASE_DIR / "outputs" / "fotomultas"
FOTOMULTAS_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}

# Índice de cámara (0, 1, 2...). Ajusta según tu PC.
CAM_INDEX = 1  # si ves la pantalla de EOS, prueba 0 o 2


def live_camera(
    plate_model_path: str | None = None,
    vehicle_model_path: str | None = None,
):
    """
    Cámara en vivo:
      - Detecta vehículos
      - Busca placa dentro del vehículo
      - OCR de la placa (formato Colombia ABC123)
      - Color del vehículo
      - SOLO aplica pico y placa y multas si la placa YA estaba en la BD
      - Guarda detecciones y actualiza la BD
    """

    init_db()

    # Modelo de PLACAS
    if plate_model_path is None:
        plate_model_path = str(BASE_DIR / "Models" / "yolov8_placas_best.pt")
        # Si ves que el viejo te funciona mejor:
        # plate_model_path = str(BASE_DIR / "Models" / "license_plate_detector.pt")

    # Modelo de VEHÍCULOS (por ejemplo YOLO entrenado en COCO)
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

    cooldown: dict[str, datetime] = {}
    COOLDOWN_SEC = 10
    video_name = "<live_camera>"

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ No se pudo leer frame de la cámara, reintentando...")
                continue

            ts = datetime.now()  # timestamp real de detección (para BD / fotomultas)
            h, w = frame.shape[:2]

            # 1) Detectar vehículos
            try:
                vr = veh_model(frame, conf=0.35, verbose=False)[0]
            except Exception as e:
                print("⚠️ Error en detección de vehículos:", e)
                cv2.imshow("Pico y Placa IA - Camara en vivo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

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
                try:
                    pr = plate_model(vehicle_crop, conf=0.25, verbose=False)[0]
                except Exception as e:
                    print("⚠️ Error en detección de placas:", e)
                    continue

                if pr.boxes is None or len(pr.boxes) == 0:
                    continue  # sin placas → siguiente vehículo

                for pbox in pr.boxes:
                    try:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])

                        # Convertir a coordenadas globales
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

                        # Caja de placa (amarillo por defecto)
                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                        label = "PLACA?"

                        # 3) OCR con try/except
                        try:
                            plate_text = ocr_placa(plate_crop)
                        except Exception as e:
                            print("⚠️ Error en OCR de placa:", e)
                            plate_text = None

                        # Filtro: formato Colombia ABC123 (6 chars)
                        if not plate_text or len(plate_text) != 6:
                            cv2.putText(
                                frame,
                                label,
                                (gx1, max(20, gy1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            continue

                        # 4) Color del vehículo
                        try:
                            color = get_vehicle_color(vehicle_crop)
                        except Exception as e:
                            print("⚠️ Error calculando color:", e)
                            color = "desconocido"

                        # 5) Consultar si YA existe en la BD
                        try:
                            st = get_plate_status(plate_text)
                            registered_before = bool(st.get("first_seen"))
                        except Exception as e:
                            print("⚠️ Error consultando estado en BD:", e)
                            registered_before = False

                        # 6) Registrar/actualizar SIEMPRE el vehículo (para futuras detecciones)
                        try:
                            upsert_vehicle(plate_text, color, ts)
                        except Exception as e:
                            print("⚠️ Error guardando vehículo en BD:", e)

                        bbox_str = f"{gx1},{gy1},{gx2},{gy2}"
                        photo_path = None

                        # =============================
                        # CASO A: NO ESTABA REGISTRADO
                        # =============================
                        if not registered_before:
                            motivo = "Vehículo no registrado previamente en la base de datos."
                            print(f"[INFO] Primera vez que se ve la placa {plate_text}. No se aplica multa.")

                            # Guardar detección sin multa
                            try:
                                insert_detection(
                                    plate_text,
                                    color,
                                    ts,
                                    video_name,
                                    bbox_str,
                                    False,          # violation
                                    motivo,
                                    None,
                                )
                            except Exception as e:
                                print("⚠️ Error guardando detección en BD:", e)

                            # Etiqueta visual especial
                            label = f"{plate_text} ({color}) NO REGISTRADO"
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)  # naranja
                            cv2.putText(
                                frame,
                                label,
                                (gx1, max(20, gy1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),
                                2,
                            )
                            # ✅ No evaluamos pico y placa ni multa para este vehículo
                            continue

                        # ==========================
                        # CASO B: YA ESTABA EN LA BD
                        # ==========================
                        # Aquí SÍ aplicamos pico y placa y multas
                        try:
                            # ⚠️ cambio clave: dejamos que el módulo de reglas
                            # decida la fecha (DEMO o real) pasando None.
                            ok, detalle = puede_circular_pasto(plate_text, None)
                            violation = not ok
                            motivo_regla = ""
                            if isinstance(detalle, dict):
                                motivo_regla = detalle.get("motivo", "")
                        except Exception as e:
                            print("⚠️ Error evaluando pico y placa:", e)
                            violation = False
                            motivo_regla = "Error al evaluar pico y placa."

                        if violation:
                            motivo_violation = (
                                "Infracción por circular en día y horario de pico y placa."
                            )
                            if motivo_regla:
                                motivo_violation += " " + motivo_regla
                        else:
                            motivo_violation = motivo_regla or "Circulación permitida."

                        if violation:
                            last_t = cooldown.get(plate_text)
                            if (not last_t) or (
                                ts - last_t
                            ) > timedelta(seconds=COOLDOWN_SEC):
                                cooldown[plate_text] = ts
                                try:
                                    photo_path = (
                                        FOTOMULTAS_DIR
                                        / f"{plate_text}_{ts:%Y%m%d_%H%M%S}.jpg"
                                    )
                                    cv2.imwrite(str(photo_path), frame)

                                    insert_violation(
                                        plate_text,
                                        color,
                                        ts,
                                        motivo_violation,
                                        str(photo_path),
                                    )
                                    print(f"[MULTA] {plate_text} - {motivo_violation}")
                                except Exception as e:
                                    print("⚠️ Error guardando fotomulta/violación:", e)

                        try:
                            insert_detection(
                                plate_text,
                                color,
                                ts,
                                video_name,
                                bbox_str,
                                violation,
                                motivo_violation,
                                str(photo_path) if photo_path else None,
                            )
                        except Exception as e:
                            print("⚠️ Error guardando detección en BD:", e)

                                                # Dibujar etiqueta final para vehículos REGISTRADOS
                        # -----------------------------------------------
                        # Queremos que el mensaje diga SOLO:
                        # - si tiene pico y placa hoy
                        # - si tiene multa
                        tiene_pico_hoy = not ok          # False -> no tiene pico; True -> sí
                        tiene_multa = violation           # True si está en infracción

                        texto_pico = "SÍ" if tiene_pico_hoy else "NO"
                        texto_multa = "SÍ" if tiene_multa else "NO"

                        # Rectángulo: rojo si hay infracción, verde si no
                        if violation:
                            cv2.rectangle(
                                frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2
                            )
                        else:
                            cv2.rectangle(
                                frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2
                            )

                        # 🔹 Etiqueta con el formato que pediste:
                        #    PLACA | Pico y placa hoy: SÍ/NO | Multa: SÍ/NO
                        label = (
                            f"{plate_text} | "
                            f"Pico y placa hoy: {texto_pico} | "
                            f"Multa: {texto_multa}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (gx1, max(20, gy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )


                    except Exception as e:
                        print("⚠️ Error procesando una placa:", e)
                        continue

            cv2.imshow("Pico y Placa IA - Camara en vivo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            print("⚠️ Error inesperado en el loop principal:", e)
            continue

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cámara detenida. Datos guardados en la base de datos.")


if __name__ == "__main__":
    live_camera()
