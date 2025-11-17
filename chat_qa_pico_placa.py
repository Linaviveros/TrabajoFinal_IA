

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import re

import pandas as pd

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


def _now() -> datetime:
    """Misma lógica de 'ahora' que en el resto del proyecto."""
    return DEMO_DATETIME if DEMO_MODE else datetime.now()


def _ultimo_digito(placa: str) -> Optional[int]:
    for ch in reversed(placa.upper()):
        if ch.isdigit():
            return int(ch)
    return None


def _dias_por_digito(digito: int) -> List[str]:
    dias: List[str] = []
    for dia, lista in RESTRICCIONES.items():
        if digito in lista:
            dias.append(dia)
    return dias


def _info_placa_bd(placa: str) -> Optional[Dict[str, Any]]:
    """
    Usa get_plate_status para traer info de la BD.
    Devuelve None si la placa nunca se ha visto.
    """
    st = get_plate_status(placa.upper())
    first_seen = st.get("first_seen")
    last_seen = st.get("last_seen")
    violations = st.get("violations", 0)

    if first_seen is None and last_seen is None and violations == 0:
        return None

    if st.get("color") is None:
        st["color"] = "DESCONOCIDO"

    return st


def _lista_placas_bd() -> Dict[str, Any]:
    """
    Devuelve un resumen de todas las placas registradas en la tabla vehicles.
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT plate, color, first_seen, last_seen, violations
                FROM vehicles
                ORDER BY last_seen DESC
                """,
                conn,
            )
    except Exception as e:
        return {
            "respuesta": f"No pude leer las placas desde la base de datos: {e}",
            "tipo": "error_bd",
            "detalle": {},
        }

    if df.empty:
        return {
            "respuesta": "Por ahora no hay placas registradas en la base de datos.",
            "tipo": "lista_placas_vacia",
            "detalle": {"placas": []},
        }

    lineas = []
    placas_detalle: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        placa = row["plate"]
        color = row["color"] or "DESCONOCIDO"
        viol = int(row["violations"] or 0)
        first_seen = row["first_seen"]
        last_seen = row["last_seen"]

        lineas.append(
            f"- {placa} ({color}), primeras detecciones: {first_seen}, "
            f"última: {last_seen}, infracciones: {viol}."
        )
        placas_detalle.append(
            {
                "plate": placa,
                "color": color,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "violations": viol,
            }
        )

    texto = "Placas registradas en el sistema:\n" + "\n".join(lineas)
    return {
        "respuesta": texto,
        "tipo": "lista_placas",
        "detalle": {"placas": placas_detalle},
    }


def _digitos_hoy() -> Dict[str, Any]:
    """
    Devuelve qué dígitos tienen pico y placa HOY, según RESTRICCIONES.
    """
    now = _now()
    fecha = now.date()
    dia_py = fecha.strftime("%A").upper()
    dia_es = DIA_MAP.get(dia_py, dia_py)

    es_fin_de_semana = dia_es in ("SABADO", "DOMINGO")
    es_festivo = fecha in FESTIVOS_2025

    if es_fin_de_semana or es_festivo:
        return {
            "respuesta": (
                f"Hoy es {dia_es} ({fecha}) y NO aplica pico y placa en Pasto "
                f"porque es fin de semana o festivo."
            ),
            "tipo": "digitos_hoy_sin_restriccion",
            "detalle": {
                "fecha": fecha.isoformat(),
                "dia": dia_es,
                "digitos_restringidos": [],
                "es_fin_de_semana": es_fin_de_semana,
                "es_festivo": es_festivo,
            },
        }

    digitos = RESTRICCIONES.get(dia_es, [])
    if not digitos:
        texto = (
            f"Hoy es {dia_es} ({fecha}) y no hay dígitos configurados con pico y placa."
        )
    else:
        dig_str = ", ".join(str(d) for d in digitos)
        texto = (
            f"Hoy es {dia_es} ({fecha}). Tienen pico y placa las placas cuyo "
            f"último dígito es: {dig_str}, entre "
            f"{HORARIO_INICIO.strftime('%H:%M')} y {HORARIO_FIN.strftime('%H:%M')}."
        )

    return {
        "respuesta": texto,
        "tipo": "digitos_hoy",
        "detalle": {
            "fecha": fecha.isoformat(),
            "dia": dia_es,
            "digitos_restringidos": digitos,
            "es_fin_de_semana": es_fin_de_semana,
            "es_festivo": es_festivo,
        },
    }


def _extraer_placa(text: str) -> Optional[str]:
    """
    Intenta extraer una placa del texto.
    Acepta cosas tipo:
      - ABC123
      - ABC 123
      - ABC-123
      - HH1228 (2–4 letras + 2–4 dígitos)
    """
    m = re.search(r"[A-Z]{2,4}\s?-?\s?\d{2,4}", text.upper())
    if not m:
        return None
    return m.group(0).replace(" ", "").replace("-", "")


# ================================
# 🌐 Función principal del chatbot
# ================================

def responder_chat(text: str) -> Dict[str, Any]:
    """
    Recibe el texto que escribió el usuario y devuelve un dict JSON:
      {
        "respuesta": "...",
        "tipo": "tipo_respuesta",
        "detalle": {...}
      }
    Esta función es la que debe llamar el endpoint /preguntar.
    """
    text = text.strip()
    text_up = text.upper()

    placa = _extraer_placa(text)

    # 1) Preguntas tipo: "¿Qué dígitos tienen pico y placa hoy?"
    if "DIGITO" in text_up and "HOY" in text_up:
        return _digitos_hoy()

    # 2) Preguntas tipo: "Muéstrame todas las placas registradas"
    if ("TODAS" in text_up or "TODOS" in text_up or "LISTA" in text_up) and "PLACA" in text_up:
        return _lista_placas_bd()

    # 3) Preguntas tipo: "¿Tengo pico y placa hoy? placa ABC123"
    if placa and "PICO" in text_up and "HOY" in text_up:
        now = _now()
        ok, detalle = puede_circular_pasto(placa, now)
        motivo = ""
        if isinstance(detalle, dict):
            motivo = detalle.get("motivo", "")
        fecha = now.date().isoformat()
        texto = (
            f"Para hoy {fecha}, la placa {placa} "
            f"{'PUEDE' if ok else 'NO PUEDE'} circular. {motivo}".strip()
        )
        return {
            "respuesta": texto,
            "tipo": "pico_hoy_placa",
            "detalle": {
                "placa": placa,
                "puede_circular": ok,
                "fecha": fecha,
                "detalle_regla": detalle,
            },
        }

    # 4) Preguntas tipo: "¿Mi placa ABC123 tiene multas / infracciones / comparendos?"
    if placa and (
        "MULTA" in text_up or "INFRACC" in text_up or "COMPAREN" in text_up
    ):
        info = _info_placa_bd(placa)
        if info is None:
            return {
                "respuesta": (
                    f"La placa {placa} no está registrada en la base de datos, "
                    f"así que no se reportan infracciones."
                ),
                "tipo": "multas_placa_no_registrada",
                "detalle": {"placa": placa},
            }

        v = info.get("violations", 0)
        color = info.get("color", "DESCONOCIDO")
        last_seen = info.get("last_seen")

        if v > 0:
            texto = (
                f"La placa {placa} ({color}) registra {v} infracción(es) vigentes por pico y placa. "
                f"Última detección en el sistema: {last_seen}."
            )
        else:
            texto = (
                f"La placa {placa} ({color}) está registrada SIN infracciones de pico y placa. "
                f"Última detección: {last_seen}."
            )

        return {
            "respuesta": texto,
            "tipo": "multas_placa",
            "detalle": info,
        }

    # 5) Preguntas tipo: "¿Cuándo tengo pico y placa? mi placa es ABC123"
    if placa and ("CUANDO" in text_up or "DIA" in text_up or "DÍA" in text):
        ultimo = _ultimo_digito(placa)
        if ultimo is None:
            return {
                "respuesta": f"No pude identificar el último dígito de la placa {placa}.",
                "tipo": "error_placa",
                "detalle": {"placa": placa},
            }

        dias = _dias_por_digito(ultimo)
        if dias:
            dias_str = ", ".join(dias)
            texto_dias = (
                f"La placa {placa} termina en {ultimo}. "
                f"Por las reglas configuradas, tiene pico y placa los días: {dias_str}."
            )
        else:
            texto_dias = (
                f"La placa {placa} termina en {ultimo}, pero ese dígito no está "
                f"configurado en las restricciones de pico y placa."
            )

        info = _info_placa_bd(placa)
        if info is None:
            texto_bd = (
                "Esta placa aún no está registrada en la base de datos de detecciones, "
                "por lo que no se reportan infracciones."
            )
            detalle_bd: Dict[str, Any] = {}
        else:
            v = info.get("violations", 0)
            color = info.get("color", "DESCONOCIDO")
            last_seen = info.get("last_seen")
            if v > 0:
                texto_bd = (
                    f"En la base de datos aparece con color {color} y {v} infracción(es) registradas. "
                    f"Última detección: {last_seen}."
                )
            else:
                texto_bd = (
                    f"En la base de datos aparece con color {color} y SIN infracciones registradas. "
                    f"Última detección: {last_seen}."
                )
            detalle_bd = info

        return {
            "respuesta": texto_dias + " " + texto_bd,
            "tipo": "info_placa_pico_y_multas",
            "detalle": {
                "placa": placa,
                "ultimo_digito": ultimo,
                "dias_restriccion": dias,
                "estado_bd": detalle_bd,
            },
        }

    # 6) Preguntas generales tipo "¿Hay pico y placa los fines de semana?"
    # → respondemos con una explicación básica de las reglas.
    if "FIN DE SEMANA" in text_up or "SABADO" in text_up or "SÁBADO" in text or "DOMINGO" in text_up:
        return {
            "respuesta": (
                "En Pasto, según la configuración de este proyecto, el pico y placa "
                "solo aplica de lunes a viernes de "
                f"{HORARIO_INICIO.strftime('%H:%M')} a {HORARIO_FIN.strftime('%H:%M')}. "
                "Los sábados, domingos y festivos NO tienen restricción."
            ),
            "tipo": "reglas_generales",
            "detalle": {},
        }

    # 7) Fallback: mensaje de ayuda
    ayuda = (
        "No entendí bien la pregunta. Puedes intentar algo como:\n"
        "- ¿Qué dígitos tienen pico y placa hoy?\n"
        "- ¿Tengo pico y placa hoy? placa ABC123\n"
        "- ¿Cuándo tengo pico y placa? mi placa es ABC123\n"
        "- ¿La placa ABC123 tiene multas?\n"
        "- Muéstrame todas las placas registradas"
    )
    return {
        "respuesta": ayuda,
        "tipo": "ayuda",
        "detalle": {},
    }


# ================================
# 🎨 HTML del chatbot (para /chatbot)
# ================================

def get_chatbot_html() -> str:
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <title>Chatbot Pico y Placa – Pasto</title>
        <style>
            :root {
                color-scheme: light dark;
            }
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                height: 100vh;
                background: #0f172a;
                color: #e5e7eb;
            }
            header {
                padding: 1rem 1.5rem;
                background: #020617;
                border-bottom: 1px solid #1f2937;
            }
            header h1 {
                margin: 0;
                font-size: 1.25rem;
            }
            header p {
                margin: 0.25rem 0 0;
                font-size: 0.85rem;
                color: #9ca3af;
            }
            .chat-container {
                flex: 1;
                display: flex;
                justify-content: center;
                padding: 1rem;
            }
            .chat-card {
                width: 100%;
                max-width: 900px;
                background: #020617;
                border-radius: 12px;
                border: 1px solid #1f2937;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-messages {
                flex: 1;
                padding: 1rem;
                overflow-y: auto;
                scroll-behavior: smooth;
            }
            .msg {
                margin-bottom: 0.75rem;
                max-width: 80%;
                padding: 0.6rem 0.8rem;
                border-radius: 10px;
                font-size: 0.9rem;
                white-space: pre-wrap;
                word-break: break-word;
            }
            .msg.user {
                margin-left: auto;
                background: #1d4ed8;
                color: white;
                border-bottom-right-radius: 2px;
            }
            .msg.bot {
                margin-right: auto;
                background: #111827;
                border-bottom-left-radius: 2px;
            }
            .meta {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-top: 0.25rem;
            }
            .input-area {
                padding: 0.75rem;
                border-top: 1px solid #1f2937;
                background: #020617;
                display: flex;
                gap: 0.5rem;
                align-items: center;
            }
            .input-area input {
                flex: 1;
                padding: 0.6rem 0.75rem;
                border-radius: 999px;
                border: 1px solid #4b5563;
                outline: none;
                background: #020617;
                color: #e5e7eb;
            }
            .input-area input:focus {
                border-color: #60a5fa;
            }
            .input-area button {
                padding: 0.55rem 1rem;
                border-radius: 999px;
                border: none;
                background: #22c55e;
                color: #022c22;
                font-weight: 600;
                cursor: pointer;
            }
            .input-area button:disabled {
                opacity: 0.5;
                cursor: default;
            }
            .toolbar {
                display: flex;
                justify-content: flex-end;
                gap: 0.5rem;
                padding: 0.5rem 1rem 0.25rem;
                border-bottom: 1px solid #1f2937;
                background: #020617;
            }
            .toolbar button {
                font-size: 0.75rem;
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                border: 1px solid #374151;
                background: #020617;
                color: #9ca3af;
                cursor: pointer;
            }
            .toolbar button:hover {
                border-color: #60a5fa;
                color: #e5e7eb;
            }
            .json-toggle {
                font-size: 0.8rem;
                color: #60a5fa;
                cursor: pointer;
                margin-top: 0.25rem;
            }
            .json-block {
                margin-top: 0.25rem;
                padding: 0.4rem 0.5rem;
                background: #020617;
                border-radius: 6px;
                border: 1px solid #1f2937;
                font-size: 0.75rem;
                max-height: 220px;
                overflow: auto;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Chatbot Pico y Placa – Pasto</h1>
            <p>Pregunta por placas, multas, color del vehículo, dígitos con pico y placa hoy, etc.</p>
        </header>

        <div class="chat-container">
            <div class="chat-card">
                <div class="toolbar">
                    <button onclick="clearChat()">Limpiar chat</button>
                    <button onclick="sendExample('¿Qué dígitos tienen pico y placa hoy?')">
                        Ejemplo: dígitos de hoy
                    </button>
                    <button onclick="sendExample('¿Cuándo tengo pico y placa? mi placa es ABC123')">
                        Ejemplo: placa
                    </button>
                </div>

                <div id="messages" class="chat-messages"></div>

                <div class="input-area">
                    <input id="user_input" type="text"
                           placeholder="Escribe tu pregunta, ej: ¿La placa ELY788 tiene multas?"
                           onkeydown="if(event.key==='Enter'){sendMessage();}" />
                    <button id="send_btn" onclick="sendMessage()">Enviar</button>
                </div>
            </div>
        </div>

        <script>
            const baseUrl = window.location.origin;
            const messagesDiv = document.getElementById("messages");
            const inputEl = document.getElementById("user_input");
            const sendBtn = document.getElementById("send_btn");

            function addMessage(text, sender="bot", meta="", jsonDetail=null) {
                const msg = document.createElement("div");
                msg.className = "msg " + sender;
                msg.textContent = text;

                const wrapper = document.createElement("div");
                wrapper.appendChild(msg);

                if (meta || jsonDetail) {
                    const metaDiv = document.createElement("div");
                    metaDiv.className = "meta";
                    metaDiv.textContent = meta;
                    wrapper.appendChild(metaDiv);
                }

                if (jsonDetail) {
                    const toggle = document.createElement("div");
                    toggle.className = "json-toggle";
                    toggle.textContent = "Ver detalle técnico (JSON) ▼";
                    const pre = document.createElement("pre");
                    pre.className = "json-block";
                    pre.textContent = JSON.stringify(jsonDetail, null, 2);
                    pre.style.display = "none";

                    toggle.onclick = () => {
                        const visible = pre.style.display === "block";
                        pre.style.display = visible ? "none" : "block";
                        toggle.textContent = visible
                            ? "Ver detalle técnico (JSON) ▼"
                            : "Ocultar detalle técnico (JSON) ▲";
                    };

                    wrapper.appendChild(toggle);
                    wrapper.appendChild(pre);
                }

                messagesDiv.appendChild(wrapper);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            async function sendMessage() {
                const text = inputEl.value.trim();
                if (!text) return;

                addMessage(text, "user");
                inputEl.value = "";
                inputEl.focus();
                sendBtn.disabled = true;

                try {
                    const url = baseUrl + "/preguntar?text=" + encodeURIComponent(text);
                    const res = await fetch(url);
                    const data = await res.json();

                    const respuesta = data.respuesta || "[Sin texto de respuesta]";
                    const tipo = data.tipo || "desconocido";

                    const meta = "Tipo de respuesta: " + tipo;
                    addMessage(respuesta, "bot", meta, data.detalle || data);
                } catch (err) {
                    console.error(err);
                    addMessage(
                        "Ocurrió un error al contactar con el servidor. " +
                        "Verifica que la API esté levantada.",
                        "bot"
                    );
                } finally {
                    sendBtn.disabled = false;
                }
            }

            function clearChat() {
                messagesDiv.innerHTML = "";
                addMessage(
                    "Hola 👋 Soy el asistente de Pico y Placa de Pasto.\\n" +
                    "Puedes preguntarme cosas como:\\n" +
                    "- ¿Qué dígitos tienen pico y placa hoy?\\n" +
                    "- ¿Cuándo tengo pico y placa? mi placa es ABC123\\n" +
                    "- ¿La placa ABC123 tiene multas?\\n" +
                    "- Muéstrame todas las placas registradas",
                    "bot"
                );
            }

            function sendExample(text) {
                inputEl.value = text;
                sendMessage();
            }

            // Mensaje inicial
            clearChat();
        </script>
    </body>
    </html>
    """  # noqa: E501
