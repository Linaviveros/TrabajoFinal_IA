import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from Reglas.pico_placa_pasto_2026 import puede_circular_pasto  # 👈 NUEVO

DB_PATH = Path("outputs") / "pico_placa.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate TEXT PRIMARY KEY,
            color TEXT,
            first_seen TEXT,
            last_seen TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            color TEXT,
            timestamp TEXT,
            source TEXT,
            bbox TEXT,
            violation INTEGER,
            reason TEXT,
            photo_path TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            color TEXT,
            timestamp TEXT,
            reason TEXT,
            photo_path TEXT
        )""")

def upsert_vehicle(plate: str, color: str, ts: datetime):
    ts_iso = ts.isoformat()
    with get_conn() as c:
        cur = c.execute("SELECT plate FROM vehicles WHERE plate = ?", (plate,))
        if cur.fetchone():
            c.execute("UPDATE vehicles SET color=?, last_seen=? WHERE plate=?",
                      (color, ts_iso, plate))
        else:
            c.execute("INSERT INTO vehicles (plate, color, first_seen, last_seen) VALUES (?,?,?,?)",
                      (plate, color, ts_iso, ts_iso))

def insert_detection(plate: str, color: str, ts: datetime, source: str, bbox: str,
                     violation: bool, reason: str, photo_path: str | None):
    with get_conn() as c:
        c.execute("""INSERT INTO detections
                    (plate,color,timestamp,source,bbox,violation,reason,photo_path)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (plate, color, ts.isoformat(), source, bbox, int(violation), reason, photo_path))

def insert_violation(plate: str, color: str, ts: datetime, reason: str, photo_path: str | None):
    with get_conn() as c:
        c.execute("""INSERT INTO violations (plate,color,timestamp,reason,photo_path)
                     VALUES (?,?,?,?,?)""",
                  (plate, color, ts.isoformat(), reason, photo_path))

def get_plate_status(plate: str) -> Dict[str, Any]:
    with get_conn() as c:
        # último color, primeras/últimas detecciones y cantidad de violaciones
        cur = c.execute(
            "SELECT color, first_seen, last_seen FROM vehicles WHERE plate=?",
            (plate,)
        )
        row = cur.fetchone()
        if row:
            color, first_seen, last_seen = row
        else:
            color, first_seen, last_seen = (None, None, None)

        cur2 = c.execute("SELECT COUNT(*) FROM violations WHERE plate=?", (plate,))
        viol_count = cur2.fetchone()[0]

        return {
            "plate": plate,
            "color": color,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "violations": viol_count,
        }

# ==========================================================
# 📌 Helper para registrar detección + multa si aplica
# ==========================================================

def registrar_deteccion_y_multa(
    plate: str,
    color: str,
    ts: datetime,
    source: str,
    bbox: str,
    photo_path: str | None = None,
) -> Dict[str, Any]:
    """
    Registra una detección de placa y, si en ese momento está en pico y placa,
    crea automáticamente una multa.

    - Normaliza la placa.
    - Actualiza/crea el vehículo en 'vehicles'.
    - Inserta fila en 'detections'.
    - Si no puede circular (pico y placa en horario) -> inserta fila en 'violations'.
    """
    # Normalizar placa
    norm_plate = plate.upper().replace(" ", "").replace("-", "")

    # Evaluar reglas de pico y placa en ese momento
    puede, detalle = puede_circular_pasto(norm_plate, ts)

    if isinstance(detalle, dict):
        motivo = detalle.get("motivo") or detalle.get("detalle") or ""
    else:
        motivo = str(detalle) if detalle is not None else ""

    es_violacion = not puede

    # Actualizar/crear vehículo
    upsert_vehicle(norm_plate, color, ts)

    # Registrar la detección completa
    insert_detection(
        norm_plate,
        color,
        ts,
        source,
        bbox,
        es_violacion,
        motivo,
        photo_path,
    )

    # Si estaba en pico y placa -> registrar multa
    if es_violacion:
        insert_violation(
            norm_plate,
            color,
            ts,
            motivo,
            photo_path,
        )

    return {
        "plate": norm_plate,
        "timestamp": ts.isoformat(),
        "es_violacion": es_violacion,
        "motivo": motivo,
        "detalle_regla": detalle,
    }
