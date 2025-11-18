"""
Módulo del chatbot inteligente con procesamiento de lenguaje natural (NLP).
Entiende preguntas en lenguaje natural y busca en la base de datos.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re

from db.db import get_conn, get_plate_status
from Reglas.pico_placa_pasto_2026 import (
    HORARIO_INICIO,
    HORARIO_FIN,
    RESTRICCIONES,
    FESTIVOS_2025,
    DIA_MAP,
    DEMO_MODE,
    DEMO_DATETIME,
    puede_circular_pasto,
)

# =====================================
# 🤖 Carga opcional del modelo de NLP
# =====================================

try:
    from transformers import pipeline as _hf_pipeline

    _qa_nlp = _hf_pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
    )
    print(
        "[CHATBOT] Modelo NLP cargado correctamente "
        "(deepset/roberta-base-squad2)."
    )
except Exception as e:
    _qa_nlp = None
    print(f"[CHATBOT] No se pudo inicializar el modelo NLP: {e}")


def _now() -> datetime:
    """Misma lógica de 'ahora' que en el resto del proyecto."""
    return DEMO_DATETIME if DEMO_MODE else datetime.now()


def _extraer_placa(text: str) -> Optional[str]:
    """Extrae una placa del texto en cualquier formato razonable.

    Soporta:
    - Formatos clásicos: ABC123, ABC-123, ABC 123
    - Variantes con letra al final: BAB89E, ABC123D
    """
    text_up = text.upper()

    patterns = [
        # 2–4 letras + opcional espacio/guion + 2–4 dígitos + opcional letra final
        r"\b[A-Z]{2,4}\s*-?\s*\d{2,4}[A-Z]?\b",
        # 3 letras + 3 dígitos + opcional letra final (ABC123 o ABC123D)
        r"\b[A-Z]{3}\d{3}[A-Z]?\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, text_up)
        if m:
            # Quitamos espacios y guiones internos
            placa = m.group(0).replace(" ", "").replace("-", "")
            return placa

    # Fallback genérico: cualquier token de 5–8 chars alfanuméricos que tenga al menos un dígito
    m = re.search(r"\b[A-Z0-9]{5,8}\b", text_up)
    if m:
        token = m.group(0)
        if any(ch.isdigit() for ch in token):
            return token

    return None


def _ultimo_digito(placa: str) -> Optional[int]:
    """Extrae el último dígito de una placa."""
    for ch in reversed(placa.upper()):
        if ch.isdigit():
            return int(ch)
    return None


def _normalizar_texto(text: str) -> str:
    """Normaliza el texto eliminando acentos y convirtiendo a mayúsculas."""
    replacements = {
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "¿": "",
        "?": "",
        "¡": "",
        "!": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.upper()


def _detectar_intencion(text: str) -> Tuple[str, float]:
    """
    Detecta la intención de la pregunta del usuario.
    Retorna: (intencion, confianza)
    """
    text_norm = _normalizar_texto(text)

    # Definir patrones de intención con palabras clave
    intenciones = {
        "consultar_digitos_hoy": [
            "QUE DIGITOS",
            "CUALES DIGITOS",
            "DIGITOS HOY",
            "QUIEN TIENE PICO HOY",
            "QUIENES TIENEN PICO HOY",
            "QUE PLACAS HOY",
            "CUALES PLACAS HOY",
            "QUE NUMEROS",
            "CUALES NUMEROS",
            "RESTRICCION HOY",
            "PICO HOY",
        ],
        "consultar_placa_hoy": [
            "PUEDO CIRCULAR",
            "TENGO PICO",
            "TIENE PICO",
            "MI PLACA",
            "CIRCULAR HOY",
            "SALIR HOY",
            "TRANSITAR HOY",
            "PICO Y PLACA HOY",
        ],
        "consultar_cuando_pico": [
            "CUANDO TENGO",
            "CUANDO TIENE",
            "QUE DIAS",
            "CUALES DIAS",
            "EN QUE DIAS",
            "CUANDO NO PUEDO",
            "CUANDO PUEDO",
            "DIAS DE RESTRICCION",
        ],
        "consultar_multas": [
            "MULTAS",
            "INFRACCIONES",
            "COMPARENDOS",
            "SANCIONES",
            "TENGO MULTAS",
            "TIENE MULTAS",
            "DEUDAS",
            "PENDIENTES",
        ],
        "consultar_color": [
            "COLOR",
            "DE QUE COLOR",
            "QUE COLOR",
        ],
        "consultar_horario": [
            "HORARIO",
            "QUE HORA",
            "A QUE HORA",
            "DESDE QUE HORA",
            "HASTA QUE HORA",
            "HORAS",
            "CUANDO EMPIEZA",
            "CUANDO TERMINA",
        ],
        "listar_placas": [
            "TODAS LAS PLACAS",
            "LISTA DE PLACAS",
            "VER PLACAS",
            "MOSTRAR PLACAS",
            "PLACAS REGISTRADAS",
            "QUE PLACAS HAY",
        ],
        "consultar_fin_semana": [
            "FIN DE SEMANA",
            "SABADO",
            "DOMINGO",
            "FESTIVO",
            "DIA FESTIVO",
            "FERIADO",
        ],
        "info_general_placa": [
            "INFORMACION",
            "INFO",
            "DATOS",
            "ESTADO",
            "REGISTRO",
        ],
    }

    mejor_intencion = "desconocida"
    mejor_score = 0.0

    for intencion, palabras_clave in intenciones.items():
        score = 0
        for palabra in palabras_clave:
            if palabra in text_norm:
                score += 1

        # Normalizar score por número de palabras clave
        score_normalizado = score / len(palabras_clave) if palabras_clave else 0.0

        if score_normalizado > mejor_score:
            mejor_score = score_normalizado
            mejor_intencion = intencion

    return mejor_intencion, mejor_score


def _ultima_infraccion_bd(placa: str) -> Optional[datetime]:
    """
    Devuelve la fecha/hora de la última infracción registrada en la tabla `violations`
    para esa placa, o None si no tiene infracciones históricas.

    Asume una tabla violations con una columna de fecha/hora llamada `ts`.
    Si en tu DB la columna se llama diferente (por ejemplo `timestamp`),
    cambia el nombre en la consulta SQL.
    """
    try:
        with get_conn() as conn:
            cur = conn.execute(
                """
                SELECT ts
                FROM violations
                WHERE plate = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (placa.upper(),),
            )
            row = cur.fetchone()
    except Exception as e:
        print(f"[CHATBOT] Error consultando última infracción en BD: {e}")
        return None

    if not row:
        return None

    ts_value = row[0]
    # Puede venir como datetime o como texto ISO
    if isinstance(ts_value, datetime):
        return ts_value

    from datetime import datetime as _dt

    try:
        return _dt.fromisoformat(str(ts_value))
    except Exception:
        # Si no se puede parsear, devolvemos None y usamos last_seen como fallback
        return None


def _buscar_en_bd(query: str) -> List[Dict[str, Any]]:
    """
    Busca en la base de datos usando una consulta flexible.
    """
    try:
        with get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT plate, color, first_seen, last_seen, violations
                FROM vehicles
                WHERE plate LIKE ? OR color LIKE ?
                ORDER BY last_seen DESC
                LIMIT 10
                """,
                (f"%{query}%", f"%{query}%"),
            )
            rows = cursor.fetchall()

            resultados: List[Dict[str, Any]] = []
            for plate, color, first_seen, last_seen, violations in rows:
                resultados.append(
                    {
                        "plate": plate,
                        "color": color or "DESCONOCIDO",
                        "first_seen": first_seen,
                        "last_seen": last_seen,
                        "violations": int(violations or 0),
                    }
                )

            return resultados
    except Exception as e:
        print(f"Error buscando en BD: {e}")
        return []


# =================================================
# 🆕 Cálculo de PRÓXIMO pico y placa para la placa
# =================================================


def _proximo_pico_placa(
    ultimo_digito: Optional[int],
    desde: datetime,
    incluir_hoy: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Busca el próximo día (a partir de 'desde') en el que la placa
    con 'ultimo_digito' tendrá pico y placa, respetando:
      - Solo lunes a viernes
      - Festivos sin pico y placa

    Retorna: {"fecha": date, "dia": "LUNES"/...} o None.
    """
    if ultimo_digito is None:
        return None

    fecha_base = desde.date()
    # Buscamos hasta 14 días hacia adelante (suficiente para el calendario que tienes)
    start_offset = 0 if incluir_hoy else 1
    for offset in range(start_offset, 15):
        dia = fecha_base + timedelta(days=offset)
        dia_py = dia.strftime("%A").upper()
        dia_es = DIA_MAP.get(dia_py, dia_py)

        # Fines de semana y festivos: no aplica pico y placa
        if dia_es in ("SABADO", "DOMINGO") or dia in FESTIVOS_2025:
            continue

        restringidos = RESTRICCIONES.get(dia_es, [])
        if ultimo_digito in restringidos:
            return {"fecha": dia, "dia": dia_es}

    return None


def _obtener_estado_completo_placa(placa: str) -> Dict[str, Any]:
    """Obtiene el estado completo de una placa."""
    plate_up = placa.upper()
    st = get_plate_status(plate_up)

    first_seen = st.get("first_seen")
    last_seen = st.get("last_seen")
    violations = st.get("violations", 0)

    # 🔁 Mensaje actualizado para placa no registrada
    if first_seen is None and last_seen is None and violations == 0:
        return {
            "plate": plate_up,
            "registrada": False,
            "mensaje": (
                "La placa no está registrada en la base de datos. "
                "Primero verifica tu carro con el sensor de cámara y luego "
                "podrás consultar si tiene multas relacionadas con pico y placa."
            ),
        }

    detectada = last_seen is not None
    if st.get("color") is None:
        st["color"] = "DESCONOCIDO"

    now = _now()
    ok, detalle = puede_circular_pasto(plate_up, now)

    fecha = now.date()
    hora_time = now.time()
    dia_py = fecha.strftime("%A").upper()
    dia_es = DIA_MAP.get(dia_py, dia_py)
    ultimo = _ultimo_digito(plate_up)
    restringidos = RESTRICCIONES.get(dia_es, [])

    es_fin_de_semana = dia_es in ("SABADO", "DOMINGO")
    es_festivo = fecha in FESTIVOS_2025

    tiene_pico_hoy = (
        not es_fin_de_semana
        and not es_festivo
        and ultimo is not None
        and ultimo in restringidos
    )

    # 🆕 Cálculo del próximo pico y placa PARA LA PLACA
    proximo_info = _proximo_pico_placa(ultimo, now)
    if proximo_info is not None:
        prox_fecha = proximo_info["fecha"]
        prox_dia = proximo_info["dia"]
        prox_str = prox_fecha.strftime("%d/%m/%Y")
        texto_proximo = f" Tu próximo pico y placa es el {prox_dia.lower()} {prox_str}."
    else:
        texto_proximo = ""

    motivo_regla = None
    if isinstance(detalle, dict):
        motivo_regla = detalle.get("motivo") or detalle.get("detalle")

    if not tiene_pico_hoy:
        # 👉 Caso: HOY NO tiene pico y placa
        base_msg = "Hoy no tienes pico y placa, circula con tranquilidad."
        mensaje_horario = base_msg + texto_proximo
        en_infraccion = False
    else:
        # 👉 Caso: HOY SÍ tiene pico y placa
        if hora_time < HORARIO_INICIO:
            base = (
                "Hoy tienes pico y placa, pero la restricción aún no empieza. "
                f"El horario va de {HORARIO_INICIO.strftime('%H:%M')} a "
                f"{HORARIO_FIN.strftime('%H:%M')}."
            )
            mensaje_horario = f"{base} {motivo_regla}" if motivo_regla else base
            if texto_proximo:
                mensaje_horario += texto_proximo
            en_infraccion = False
        elif hora_time > HORARIO_FIN:
            base = (
                "Hoy estabas de pico y placa, pero la restricción ya terminó "
                f"porque son después de las {HORARIO_FIN.strftime('%H:%M')}."
            )
            mensaje_horario = f"{base} {motivo_regla}" if motivo_regla else base
            if texto_proximo:
                mensaje_horario += texto_proximo
            en_infraccion = False
        else:
            if detectada:
                base = (
                    "⚠️ En este momento estás en horario de pico y placa y el vehículo "
                    "ha sido detectado circulando: estás violando la normativa."
                )
                mensaje_horario = f"{base} {motivo_regla}" if motivo_regla else base
                if texto_proximo:
                    mensaje_horario += texto_proximo
                en_infraccion = True
            else:
                base = (
                    "En este momento estás en horario de pico y placa. "
                    "Si el vehículo circula, estaría violando la normativa."
                )
                mensaje_horario = f"{base} {motivo_regla}" if motivo_regla else base
                if texto_proximo:
                    mensaje_horario += texto_proximo
                en_infraccion = False

    if tiene_pico_hoy:
        motivo_pico = (
            f"Hoy la placa {plate_up} TIENE pico y placa (último dígito {ultimo}). "
            f"No puede circular entre {HORARIO_INICIO.strftime('%H:%M')} y "
            f"{HORARIO_FIN.strftime('%H:%M')}."
        )
    else:
        if es_fin_de_semana or es_festivo:
            motivo_pico = "Hoy no aplica pico y placa porque es fin de semana o festivo."
        else:
            motivo_pico = (
                f"Hoy la placa {plate_up} NO tiene pico y placa "
                f"(último dígito {ultimo})."
            )

    return {
        **st,
        "detectada": detectada,
        "consulta": now.isoformat(),
        "puede_circular_hoy": ok,
        "pico_placa_detalle": {
            "tiene_pico_hoy": tiene_pico_hoy,
            "motivo": motivo_pico,
        },
        "mensaje_horario": mensaje_horario,
        "en_infraccion": en_infraccion,
        # 🆕 Info técnica adicional por si la quieres usar en el front
        "proximo_pico": proximo_info,
    }


# =======================================================
# 🆕 Flujo simple: usuario solo digita la placa en el chat
# =======================================================


def _respuesta_placa_simple(placa: str) -> Dict[str, Any]:
    """
    Respuesta simplificada para cuando el usuario solo digita la placa.
    - Indica si hoy tiene pico y placa.
    - Muestra estado de multas (cantidad y fecha de la última, si existe).
    - Maneja el caso de placa no registrada.
    """
    placa_up = placa.upper()
    estado = _obtener_estado_completo_placa(placa_up)

    # Placa NO registrada en la BD
    if not estado.get("registrada", True):
        mensaje = (
            f"🔍 La placa {placa_up} no está en la base de datos.\n\n"
            "Primero verifica tu carro con el sensor de cámara y luego "
            "podrás consultar si tiene multas relacionadas con pico y placa."
        )
        return {
            "respuesta": mensaje,
            "tipo": "placa_no_registrada",
            "detalle": estado,
        }

    color = estado.get("color", "DESCONOCIDO")
    motivo_pico = estado.get("pico_placa_detalle", {}).get("motivo", "")
    mensaje_horario = estado.get("mensaje_horario", "")
    violations = int(estado.get("violations", 0) or 0)

    # 🟥 ¿Está en infracción AHORA MISMO según las reglas?
    en_infraccion = bool(estado.get("en_infraccion"))
    hoy_str = _now().strftime("%d/%m/%Y")

    if en_infraccion:
        texto_estado_actual = (
            "⚠️ En este momento el sistema indica que el vehículo "
            "ESTÁ EN INFRACCIÓN de pico y placa (circulando en horario restringido). "
            f"Esta infracción corresponde al día {hoy_str}."
        )
    else:
        texto_estado_actual = (
            "En este momento no estás en infracción de pico y placa."
        )

    # 🔎 Buscar la última infracción HISTÓRICA en la tabla violations
    ultima_infr_dt = _ultima_infraccion_bd(placa_up)
    fecha_ultima_str = ultima_infr_dt.strftime("%d/%m/%Y") if ultima_infr_dt else None

    # Construir texto de multas/historial
    if violations > 0 and fecha_ultima_str:
        texto_multas = (
            f"Tienes {violations} infracción(es) registrada(s) en el historial. "
            f"La última fue el día {fecha_ultima_str}."
        )
    elif violations > 0:
        texto_multas = (
            f"Tienes {violations} infracción(es) registrada(s) por pico y placa."
        )
    else:
        if en_infraccion:
            # Caso típico: cámara detectó infracción hoy por primera vez
            texto_multas = (
                "Esta es tu primera infracción de pico y placa detectada hoy. "
                "Aún no hay infracciones históricas registradas en la base de datos."
            )
        else:
            texto_multas = (
                "Actualmente no tienes infracciones históricas registradas en el sistema."
            )

    respuesta = (
        f"🚗 Placa {placa_up} ({color})\n\n"
        f"{motivo_pico}\n\n"
        f"📍 {mensaje_horario}\n\n"
        f"🔎 {texto_estado_actual}\n\n"
        f"📊 {texto_multas}\n\n"
        "Si quieres más detalle, puedes preguntarme por ejemplo:\n"
        f"\"¿Tengo multas con la placa {placa_up}?\""
    )

    return {
        "respuesta": respuesta,
        "tipo": "consulta_placa_simple",
        "detalle": estado,
    }


# ======================================
# 🧩 Construir contexto libre desde la BD
# ======================================


def _construir_contexto_desde_bd(max_registros: int = 200) -> str:
    """
    Convierte la tabla 'vehicles' a un texto descriptivo para que
    el modelo de NLP pueda responder preguntas libres.
    """
    try:
        with get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT plate, color, first_seen, last_seen, violations
                FROM vehicles
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (max_registros,),
            )
            rows = cursor.fetchall()
    except Exception as e:
        print(f"[CHATBOT] Error leyendo BD para contexto NLP: {e}")
        return "No hay información disponible en la base de datos de pico y placa."

    if not rows:
        return "No hay vehículos registrados en la base de datos de pico y placa."

    lineas: List[str] = []
    for plate, color, first_seen, last_seen, violations in rows:
        color = color or "desconocido"
        viol = violations or 0
        lineas.append(
            f"Vehículo con placa {plate}, color {color}, "
            f"primera detección {first_seen}, última detección {last_seen}, "
            f"número de infracciones {viol}."
        )

    return "\n".join(lineas)


# ======================================
# 🧠 LÓGICA ORIGINAL BASADA EN REGLAS
# ======================================


def _responder_chat_reglas(text: str) -> Dict[str, Any]:
    """
    Lógica original con intenciones y reglas (la que ya tenías).
    """
    text = text.strip()
    if not text:
        return {
            "respuesta": "Por favor escribe una pregunta.",
            "tipo": "error",
            "detalle": {},
        }

    # 1. Detectar intención
    intencion, confianza = _detectar_intencion(text)
    placa = _extraer_placa(text)

    print(f"[CHATBOT] Intención detectada: {intencion} (confianza: {confianza:.2f})")
    if placa:
        print(f"[CHATBOT] Placa detectada: {placa}")

    # 2. SALUDOS
    text_up = _normalizar_texto(text)
    if any(word in text_up for word in ["HOLA", "BUENOS", "BUENAS", "HEY"]):
        return {
            "respuesta": (
                "👋 ¡Hola! Soy tu asistente de Pico y Placa de Pasto.\n\n"
                "Para empezar puedes escribirme solo tu placa, por ejemplo: ABC123.\n\n"
            ),
            "tipo": "saludo",
            "detalle": {"intencion": intencion, "confianza": confianza},
        }

    # 3. CONSULTAR DÍGITOS HOY
    if intencion == "consultar_digitos_hoy":
        now = _now()
        fecha = now.date()
        dia_py = fecha.strftime("%A").upper()
        dia_es = DIA_MAP.get(dia_py, dia_py)

        es_fin_de_semana = dia_es in ("SABADO", "DOMINGO")
        es_festivo = fecha in FESTIVOS_2025

        if es_fin_de_semana or es_festivo:
            return {
                "respuesta": (
                    f"🟢 Hoy es {dia_es.lower()} ({fecha.strftime('%d/%m/%Y')}) "
                    "y NO aplica pico y placa en Pasto. ¡Puedes circular libremente!"
                ),
                "tipo": "digitos_hoy_sin_restriccion",
                "detalle": {"fecha": fecha.isoformat(), "dia": dia_es},
            }

        digitos = RESTRICCIONES.get(dia_es, [])
        if digitos:
            dig_str = ", ".join(str(d) for d in digitos)
            return {
                "respuesta": (
                    f"🚗 Hoy {dia_es.lower()} ({fecha.strftime('%d/%m/%Y')}):\n\n"
                    f"⛔ Placas con último dígito: {dig_str}\n"
                    f"⏰ Horario: {HORARIO_INICIO.strftime('%H:%M')} "
                    f"a {HORARIO_FIN.strftime('%H:%M')}"
                ),
                "tipo": "digitos_hoy",
                "detalle": {"digitos": digitos, "dia": dia_es},
            }

    # 4. CONSULTAR PLACA HOY
    if intencion == "consultar_placa_hoy" or (placa and "HOY" in text_up):
        if not placa:
            return {
                "respuesta": (
                    "Para consultar tu placa hoy, por favor indícame el número. "
                    "Ejemplo: 'ABC123'"
                ),
                "tipo": "solicitar_placa",
                "detalle": {},
            }

        estado = _obtener_estado_completo_placa(placa)

        if not estado.get("registrada", True):
            ultimo = _ultimo_digito(placa)
            if ultimo:
                now = _now()
                ok, detalle = puede_circular_pasto(placa, now)
                motivo = (
                    detalle.get("motivo", "")
                    if isinstance(detalle, dict)
                    else ""
                )

                return {
                    "respuesta": (
                        f"🔍 La placa {placa} no está en nuestra base de datos.\n\n"
                        "Primero debes verificar tu vehículo con el sensor de cámara "
                        "para que quede registrado en el sistema.\n"
                        "Después de que el sistema detecte la placa podrás consultar "
                        "si tienes multas o infracciones por pico y placa.\n\n"
                        f"Según las reglas actuales de pico y placa para ese número de placa:\n"
                        f"{motivo}"
                    ),
                    "tipo": "placa_no_registrada",
                    "detalle": {"placa": placa, "puede_circular": ok},
                }

        color = estado.get("color", "DESCONOCIDO")
        motivo = estado.get("pico_placa_detalle", {}).get("motivo", "")
        mensaje_horario = estado.get("mensaje_horario", "")
        violations = estado.get("violations", 0)

        # 🔴 NUEVO: leer si está en infracción justo ahora
        en_infraccion = bool(estado.get("en_infraccion"))
        # Fecha de hoy formateada para el mensaje
        fecha_hoy_str = _now().strftime("%d/%m/%Y")

        if en_infraccion:
            linea_infraccion = (
                "⚠️ En este momento el sistema indica que el vehículo "
                "ESTÁ EN INFRACCIÓN de pico y placa (detectado circulando en horario restringido). "
                f"Esta infracción corresponde al día {fecha_hoy_str}."
            )
        else:
            linea_infraccion = (
                "En este momento no estás en infracción de pico y placa."
            )

        respuesta = (
            f"🚗 Placa {placa} ({color})\n\n"
            f"{motivo}\n\n"
            f"📍 {mensaje_horario}\n\n"
            f"{linea_infraccion}\n\n"
            f"📊 Infracciones acumuladas en el sistema: {violations}"
        )

        return {
            "respuesta": respuesta,
            "tipo": "consulta_placa_hoy",
            "detalle": estado,
        }

    # 5. CONSULTAR CUÁNDO TIENE PICO (DÍAS)
    if intencion == "consultar_cuando_pico" or (
        placa and any(w in text_up for w in ["CUANDO", "QUE DIAS", "CUALES DIAS"])
    ):
        if not placa:
            return {
                "respuesta": (
                    "Por favor indícame la placa para consultar sus días "
                    "de restricción. Ejemplo: 'ABC123'"
                ),
                "tipo": "solicitar_placa",
                "detalle": {},
            }

        ultimo = _ultimo_digito(placa)
        if not ultimo:
            return {
                "respuesta": f"No pude identificar el último dígito de {placa}.",
                "tipo": "error",
                "detalle": {},
            }

        dias = [dia for dia, digs in RESTRICCIONES.items() if ultimo in digs]

        if dias:
            dias_str = ", ".join(dias).lower()
            return {
                "respuesta": (
                    f"🚗 Placa {placa} (último dígito: {ultimo})\n\n"
                    f"📅 Días con pico y placa: {dias_str}\n"
                    f"⏰ Horario: {HORARIO_INICIO.strftime('%H:%M')} - "
                    f"{HORARIO_FIN.strftime('%H:%M')}"
                ),
                "tipo": "cuando_pico",
                "detalle": {
                    "placa": placa,
                    "ultimo_digito": ultimo,
                    "dias": dias,
                },
            }
        else:
            return {
                "respuesta": (
                    f"🟢 La placa {placa} (dígito {ultimo}) NO tiene "
                    "restricción de pico y placa."
                ),
                "tipo": "sin_restriccion",
                "detalle": {"placa": placa, "ultimo_digito": ultimo},
            }

    # 6. CONSULTAR MULTAS
    if intencion == "consultar_multas" or (
        placa
        and any(w in text_up for w in ["MULTA", "INFRACCION", "COMPARENDO"])
    ):
        if not placa:
            # Buscar placas con multas en BD
            with get_conn() as conn:
                cursor = conn.execute(
                    "SELECT plate, violations FROM vehicles "
                    "WHERE violations > 0 "
                    "ORDER BY violations DESC "
                    "LIMIT 5"
                )
                rows = cursor.fetchall()

            if rows:
                lineas = ["📋 Placas con infracciones registradas:\n"]
                for plate_, viol in rows:
                    lineas.append(f"⚠️ {plate_}: {viol} infracción(es)")
                return {
                    "respuesta": "\n".join(lineas),
                    "tipo": "lista_multas",
                    "detalle": {
                        "placas": [
                            {"plate": p, "violations": v} for p, v in rows
                        ]
                    },
                }
            else:
                return {
                    "respuesta": "✅ No hay placas con infracciones registradas.",
                    "tipo": "sin_multas",
                    "detalle": {},
                }

        info = get_plate_status(placa)
        if not info or (
            not info.get("first_seen") and not info.get("last_seen")
        ):
            return {
                "respuesta": (
                    f"🔍 La placa {placa} no está registrada en la base de datos.\n\n"
                    "Primero verifica tu carro con el sensor de cámara y luego "
                    "podrás consultar si tiene multas relacionadas con pico y placa."
                ),
                "tipo": "placa_no_registrada",
                "detalle": {},
            }

        v = int(info.get("violations", 0) or 0)
        last_seen = info.get("last_seen")

        if v > 0:
            if last_seen:
                detalle_fecha = f" La última infracción se registró el {last_seen}."
            else:
                detalle_fecha = ""
            return {
                "respuesta": (
                    f"⚠️ Placa {placa}: {v} infracción(es) registradas "
                    f"por violar el pico y placa.{detalle_fecha}"
                ),
                "tipo": "con_multas",
                "detalle": info,
            }
        else:
            return {
                "respuesta": f"✅ Placa {placa}: Sin infracciones registradas.",
                "tipo": "sin_multas",
                "detalle": info,
            }

    # 7. CONSULTAR COLOR
    if intencion == "consultar_color" or (placa and "COLOR" in text_up):
        if not placa:
            return {
                "respuesta": (
                    "Por favor indícame la placa para consultar su color."
                ),
                "tipo": "solicitar_placa",
                "detalle": {},
            }

        info = get_plate_status(placa)
        if not info or (
            not info.get("first_seen") and not info.get("last_seen")
        ):
            return {
                "respuesta": f"🔍 No tengo información de color para {placa}.",
                "tipo": "placa_no_registrada",
                "detalle": {},
            }

        color = info.get("color") or "DESCONOCIDO"
        return {
            "respuesta": f"🎨 Placa {placa}: Color {color}",
            "tipo": "color",
            "detalle": info,
        }

    # 8. LISTAR PLACAS
    if intencion == "listar_placas":
        with get_conn() as conn:
            cursor = conn.execute(
                "SELECT plate, color, violations "
                "FROM vehicles "
                "ORDER BY last_seen DESC "
                "LIMIT 10"
            )
            rows = cursor.fetchall()

        if not rows:
            return {
                "respuesta": "📋 No hay placas registradas aún.",
                "tipo": "lista_vacia",
                "detalle": {},
            }

        lineas = ["📋 Placas registradas:\n"]
        for plate_, color_, viol in rows:
            emoji = "⚠️" if viol > 0 else "✅"
            lineas.append(
                f"{emoji} {plate_} ({color_ or 'N/A'}) - {viol} infracción(es)"
            )

        return {
            "respuesta": "\n".join(lineas),
            "tipo": "lista_placas",
            "detalle": {"total": len(rows)},
        }

    # 9. HORARIO
    if intencion == "consultar_horario":
        return {
            "respuesta": (
                "⏰ Horario de pico y placa en Pasto:\n\n"
                f"🕐 {HORARIO_INICIO.strftime('%H:%M')} a "
                f"{HORARIO_FIN.strftime('%H:%M')}\n"
                "📅 Lunes a viernes\n"
                "🟢 Fines de semana y festivos: SIN restricción"
            ),
            "tipo": "horario",
            "detalle": {},
        }

    # 10. FIN DE SEMANA
    if intencion == "consultar_fin_semana":
        return {
            "respuesta": (
                "🟢 Los fines de semana y festivos NO hay pico y placa en Pasto.\n\n"
                "Puedes circular libremente sábados, domingos y días festivos."
            ),
            "tipo": "fin_semana",
            "detalle": {},
        }

    # 11. BÚSQUEDA GENERAL EN BD POR PLACA
    if placa:
        info = get_plate_status(placa)
        if info and (info.get("first_seen") or info.get("last_seen")):
            color = info.get("color") or "DESCONOCIDO"
            viol = info.get("violations", 0)
            ultimo = _ultimo_digito(placa)
            dias = [
                dia
                for dia, digs in RESTRICCIONES.items()
                if ultimo and ultimo in digs
            ]
            dias_str = ", ".join(dias).lower() if dias else "ninguno"

            return {
                "respuesta": (
                    f"🚗 Información de {placa}:\n\n"
                    f"🎨 Color: {color}\n"
                    f"📊 Infracciones: {viol}\n"
                    f"📅 Días con pico: {dias_str}\n"
                    f"🕐 Última detección: {info.get('last_seen', 'N/A')}"
                ),
                "tipo": "info_general",
                "detalle": info,
            }

    # 12. FALLBACK REGLAS: NO ENTENDIÓ
    return {
        "respuesta": (
            "❓ No entendí bien tu pregunta.\n\n"
            "Para empezar, puedes escribirme solo tu placa, por ejemplo: ABC123.\n\n"
            "También puedo ayudarte con:\n"
            "• Consultas de pico y placa hoy\n"
            "• Placas registradas\n"
            "• Multas / infracciones\n"
            "• Horarios y días de restricción\n"
            "• Información de una placa específica\n\n"
            "Ejemplos:\n"
            "• ABC123\n"
            "• ¿Tengo multas con la placa ABC123?\n"
            "• ¿Qué dígitos tienen pico y placa hoy?"
        ),
        "tipo": "no_entendido_reglas",
        "detalle": {"intencion_detectada": intencion, "confianza": confianza},
    }


# ======================================
# 🎯 FUNCIÓN PRINCIPAL CON IA + REGLAS
# ======================================


def responder_chat(text: str) -> Dict[str, Any]:
    """
    Chatbot principal que se expone a FastAPI.

    1) Intenta responder con IA (NLP) usando como contexto la base de datos
    de vehículos (sin quemar preguntas).
    2) Si el modelo no está disponible o la confianza es baja, usa la lógica
    de reglas original (_responder_chat_reglas).
    """
    text = (text or "").strip()
    if not text:
        return {
            "respuesta": "Por favor escribe una pregunta.",
            "tipo": "error",
            "detalle": {},
        }

    # 🆕 Caso especial: el usuario solo digitó la placa (ej: "ABC123")
    placa_detectada = _extraer_placa(text)
    if placa_detectada and text.strip().upper() == placa_detectada.upper():
        return _respuesta_placa_simple(placa_detectada)

    # 1) IA sobre la base de datos
    if _qa_nlp is not None:
        contexto = _construir_contexto_desde_bd()
        try:
            resultado = _qa_nlp(question=text, context=contexto)
            score = float(resultado.get("score", 0.0))
            answer = (resultado.get("answer") or "").strip()
            print(f"[CHATBOT][NLP] score={score:.3f}, answer={answer!r}")
        except Exception as e:
            print(f"[CHATBOT] Error ejecutando modelo NLP: {e}")
            score = 0.0
            answer = ""

        # Umbral de confianza para considerar la respuesta válida
        UMBRAL = 0.15
        if answer and score >= UMBRAL:
            return {
                "respuesta": answer,
                "tipo": "qa_bd",
                "detalle": {
                    "score": score,
                    "raw": resultado,
                },
            }

    # 2) Si la IA no sabe o no está disponible, usamos tus reglas
    return _responder_chat_reglas(text)


__all__ = ["responder_chat"]
