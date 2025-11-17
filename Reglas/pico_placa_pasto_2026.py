"""
Pico y Placa Pasto (Nariño) – Año 2025 (ejemplo académico)
Horario: 7:30 a.m. – 7:00 p.m.
Días: Lunes–Viernes
Sábados, domingos y festivos nacionales: no aplica restricción
"""

from datetime import datetime, date, time


HORARIO_INICIO = time(7, 30)
HORARIO_FIN = time(19,  0)

# ---------------------------------------------------------
#  Mapa base por día de la semana (se usa como "default")
#  pero puede ser SOBREESCRITO por el calendario especial
#  de noviembre–diciembre 2025.
# ---------------------------------------------------------
RESTRICCIONES = {
    "LUNES":     [0, 1],
    "MARTES":    [2, 3],
    "MIERCOLES": [4, 5],
    "JUEVES":    [6, 7],
    "VIERNES":   [8, 9],
}


DIA_MAP = {
    "MONDAY":    "LUNES",
    "TUESDAY":   "MARTES",
    "WEDNESDAY": "MIERCOLES",
    "THURSDAY":  "JUEVES",
    "FRIDAY":    "VIERNES",
    "SATURDAY":  "SABADO",
    "SUNDAY":    "DOMINGO",
}


FESTIVOS_2025 = {
    date(2025, 1,  1),  # Año Nuevo
    date(2025, 1,  6),  # Reyes Magos
    date(2025, 3, 24),  # San José
    date(2025, 4, 17),  # Jueves Santo
    date(2025, 4, 18),  # Viernes Santo
    date(2025, 5,  1),  # Día del trabajo
    date(2025, 6,  2),  # Ascensión
    date(2025, 6, 23),  # Corpus Christi
    date(2025, 6, 30),  # Sagrado Corazón / San Pedro y San Pablo
    date(2025, 7, 20),  # Día independencia
    date(2025, 8,  7),  # Batalla Boyacá
    date(2025, 8, 18),  # Asunción de la Virgen
    date(2025, 10, 13), # Día de la Raza
    date(2025, 11, 3),  # Todos los Santos
    date(2025, 11, 17), # Independencia Cartagena
    date(2025, 12, 8),  # Inmaculada Concepción
    date(2025, 12, 25), # Navidad
}

DEMO_MODE = False

DEMO_DATETIME = datetime(2025, 11, 12, 8, 0, 0)

# ---------------------------------------------------------
#  Calendario especial NOV–DIC 2025 (lo que tú enviaste)
#  Si una fecha aparece aquí, SUS DÍGITOS MANDAN
#  sobre el mapa base RESTRICCIONES.
#  None o ausencia -> "no aplica" (se maneja por fin de
#  semana / festivo como ya estaba).
# ---------------------------------------------------------
CALENDARIO_PICO_PLACA_2025: dict[date, list[int]] = {
    # NOVIEMBRE 2025
    date(2025, 11, 18): [2, 3],
    date(2025, 11, 19): [4, 5],
    date(2025, 11, 20): [6, 7],
    date(2025, 11, 21): [8, 9],

    date(2025, 11, 24): [2, 3],
    date(2025, 11, 25): [4, 5],
    date(2025, 11, 26): [6, 7],
    date(2025, 11, 27): [8, 9],
    date(2025, 11, 28): [0, 1],

    # DICIEMBRE 2025
    date(2025, 12,  1): [4, 5],
    date(2025, 12,  2): [6, 7],
    date(2025, 12,  3): [8, 9],
    date(2025, 12,  4): [0, 1],
    date(2025, 12,  5): [2, 3],

    date(2025, 12,  9): [8, 9],
    date(2025, 12, 10): [0, 1],
    date(2025, 12, 11): [2, 3],
    date(2025, 12, 12): [4, 5],

    date(2025, 12, 15): [8, 9],
    date(2025, 12, 16): [0, 1],
}


def _get_fecha_hora(fecha_hora: datetime | None) -> datetime:
    """
    Centraliza cómo se obtiene la fecha/hora:
      - Si DEMO_MODE: usa DEMO_DATETIME
      - Si no: usa la que llega o datetime.now() si viene None
    """
    if DEMO_MODE:
        return DEMO_DATETIME
    if fecha_hora is None:
        return datetime.now()
    return fecha_hora


def _ultimo_digito(placa: str) -> int:
    """Extrae el último dígito de la placa (soporta BAB89E, etc.)."""
    for ch in reversed(placa.upper()):
        if ch.isdigit():
            return int(ch)
    raise ValueError(f"Placa inválida: {placa}")


def puede_circular_pasto(placa: str, fecha_hora: datetime | None):
    """
    Determina si una placa PUEDE circular en Pasto (configuración 2025).
    Retorna: (bool, detalle_dict)

    bool:
        True  -> puede circular
        False -> NO puede circular (pico y placa dentro de horario)

    detalle_dict:
        {"motivo": "..."} con la explicación.
    """
    # 👇 aquí se aplica el “truco” de fecha fija si DEMO_MODE=True
    fecha_hora = _get_fecha_hora(fecha_hora)

    fecha = fecha_hora.date()
    hora = fecha_hora.time()

    dia_es = DIA_MAP[fecha.strftime("%A").upper()]

    # --- DÍAS SIN PICO Y PLACA (fin de semana o festivo nacional) ---
    if dia_es in ("SABADO", "DOMINGO") or fecha in FESTIVOS_2025:
        return True, {
            "motivo": f"El día {dia_es} ({fecha}) NO tiene pico y placa en Pasto."
        }

    # --- DÍGITOS RESTRINGIDOS: base por día + override del calendario ---
    ultimo = _ultimo_digito(placa)

    # mapa base por día de la semana
    restringidos = RESTRICCIONES.get(dia_es, [])

    # si la fecha está en el calendario especial, se sobreescribe
    if fecha in CALENDARIO_PICO_PLACA_2025:
        restringidos = CALENDARIO_PICO_PLACA_2025[fecha]

    en_horario = HORARIO_INICIO <= hora <= HORARIO_FIN

    if ultimo in restringidos and en_horario:
        return False, {
            "motivo": (
                f"La placa {placa} termina en {ultimo}. "
                f"No puede circular el {dia_es} entre {HORARIO_INICIO} y {HORARIO_FIN}."
            )
        }

    return True, {
        "motivo": (
            f"La placa {placa} puede circular el {dia_es} "
            f"porque está fuera de restricción o fuera del horario."
        )
    }
