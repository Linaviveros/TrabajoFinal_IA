from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, date

from fastapi.staticfiles import StaticFiles

from db.db import get_plate_status, init_db, get_conn
from Reglas.pico_placa_pasto_2026 import (
    puede_circular_pasto,
    RESTRICCIONES,
    DIA_MAP,
    HORARIO_INICIO,
    HORARIO_FIN,
    FESTIVOS_2025,
    DEMO_MODE,
    DEMO_DATETIME,
)

# 👇 Import correcto del chatbot, sin circular import
from chatbot.chatbot import responder_chat


app = FastAPI(title="Chat Pico y Placa – Pasto 2025 (Demo Pasto)", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔧 CONFIGURAR CORS PARA PERMITIR PETICIONES DESDE EL NAVEGADOR
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)


class AskPlaca(BaseModel):
    plate: str


def _now() -> datetime:
    """
    Devuelve la fecha/hora 'actual' para TODA la API.
    """
    if DEMO_MODE:
        return DEMO_DATETIME
    return datetime.now()


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    return {
        "msg": "Endpoints: /hoy, /hoy_reglas, /placa/{plate}/status, /preguntar?text=..., /chatbot, /web",
        "demo_mode": DEMO_MODE,
        "demo_datetime": DEMO_DATETIME.isoformat() if DEMO_MODE else None,
    }


@app.get("/hoy")
def hoy(plate: str | None = None):
    now = _now()
    if not plate:
        return {
            "now": now.isoformat(),
            "info": "Proporciona ?plate=ABC123 para evaluar tu placa ahora",
        }
    ok, detalle = puede_circular_pasto(plate, now)
    return {
        "plate": plate,
        "timestamp": now.isoformat(),
        "puede_circular": ok,
        "detalle": detalle,
    }


@app.get("/hoy_reglas")
def hoy_reglas():
    now = _now()
    fecha: date = now.date()

    dia_py = fecha.strftime("%A").upper()
    dia_es = DIA_MAP.get(dia_py, dia_py)

    # 👉 Fallback de días por si DIA_MAP está incompleto (solo detectaba miércoles)
    if dia_es not in RESTRICCIONES:
        _fallback_dias = {
            "MONDAY": "LUNES",
            "TUESDAY": "MARTES",
            "WEDNESDAY": "MIERCOLES",
            "THURSDAY": "JUEVES",
            "FRIDAY": "VIERNES",
            "SATURDAY": "SABADO",
            "SUNDAY": "DOMINGO",
        }
        dia_es = _fallback_dias.get(dia_py, dia_es)

    es_fin_de_semana = dia_es in ("SABADO", "DOMINGO")
    es_festivo = fecha in FESTIVOS_2025

    if es_fin_de_semana or es_festivo:
        return {
            "fecha": fecha.isoformat(),
            "dia": dia_es,
            "es_festivo": es_festivo,
            "aplica_pico_placa": False,
            "digitos_restringidos": [],
            "horario_inicio": None,
            "horario_fin": None,
            "mensaje": "Hoy no aplica pico y placa en Pasto (fin de semana o festivo).",
        }

    digitos = RESTRICCIONES.get(dia_es, [])

    if not digitos:
        msg = f"Hoy es {dia_es} y no hay dígitos restringidos configurados."
    else:
        msg = (
            f"Hoy es {dia_es}. Tienen pico y placa las placas que terminan en "
            f"{digitos} entre {HORARIO_INICIO.strftime('%H:%M')} y {HORARIO_FIN.strftime('%H:%M')}."
        )

    return {
        "fecha": fecha.isoformat(),
        "dia": dia_es,
        "es_festivo": es_festivo,
        "aplica_pico_placa": bool(digitos),
        "digitos_restringidos": digitos,
        "horario_inicio": HORARIO_INICIO.strftime("%H:%M"),
        "horario_fin": HORARIO_FIN.strftime("%H:%M"),
        "mensaje": msg,
    }


def _ultimo_digito_local(placa: str) -> int | None:
    for ch in reversed(placa.upper()):
        if ch.isdigit():
            return int(ch)
    return None


@app.get("/placa/{plate}/status")
def placa_status(plate: str):
    plate_up = plate.upper()

    st = get_plate_status(plate_up)

    first_seen = st.get("first_seen")
    last_seen = st.get("last_seen")
    violations = st.get("violations", 0)

    if first_seen is None and last_seen is None and violations == 0:
        return {
            "plate": plate_up,
            "registrada": False,
            "mensaje": "La placa no está registrada en la base de datos (nunca ha sido detectada por el sistema).",
        }

    detectada = last_seen is not None

    if st.get("color") is None:
        st["color"] = "DESCONOCIDO"

    now = _now()
    ok, detalle = puede_circular_pasto(plate_up, now)

    fecha: date = now.date()
    hora_time = now.time()

    dia_py = fecha.strftime("%A").upper()
    dia_es = DIA_MAP.get(dia_py, dia_py)

    # 👉 Fallback de días aquí también
    if dia_es not in RESTRICCIONES:
        _fallback_dias = {
            "MONDAY": "LUNES",
            "TUESDAY": "MARTES",
            "WEDNESDAY": "MIERCOLES",
            "THURSDAY": "JUEVES",
            "FRIDAY": "VIERNES",
            "SATURDAY": "SABADO",
            "SUNDAY": "DOMINGO",
        }
        dia_es = _fallback_dias.get(dia_py, dia_es)

    ultimo = _ultimo_digito_local(plate_up)
    restringidos = RESTRICCIONES.get(dia_es, [])

    es_fin_de_semana = dia_es in ("SABADO", "DOMINGO")
    es_festivo = fecha in FESTIVOS_2025

    tiene_pico_hoy = (
        not es_fin_de_semana
        and not es_festivo
        and ultimo is not None
        and ultimo in restringidos
    )

    motivo_regla = None
    if isinstance(detalle, dict):
        motivo_regla = detalle.get("motivo") or detalle.get("detalle")

    if not tiene_pico_hoy:
        mensaje_horario = "Hoy no tienes pico y placa, circula con tranquilidad."
        en_infraccion = False
    else:
        if hora_time < HORARIO_INICIO:
            base = (
                "Hoy tienes pico y placa, pero la restricción aún no empieza. "
                f"El horario va de {HORARIO_INICIO.strftime('%H:%M')} a "
                f"{HORARIO_FIN.strftime('%H:%M')}."
            )
            if motivo_regla:
                mensaje_horario = f"{base} Detalle: {motivo_regla}"
            else:
                mensaje_horario = base
            en_infraccion = False

        elif hora_time > HORARIO_FIN:
            base = (
                "Hoy estabas de pico y placa, pero la restricción ya terminó "
                f"porque son después de las {HORARIO_FIN.strftime('%H:%M')}."
            )
            if motivo_regla:
                mensaje_horario = f"{base} Detalle: {motivo_regla}"
            else:
                mensaje_horario = base
            en_infraccion = False

        else:
            if detectada:
                base = (
                    "En este momento estás en horario de pico y placa y el vehículo "
                    "ha sido detectado circulando: estás violando la normativa de pico y placa."
                )
                if motivo_regla:
                    mensaje_horario = f"{base} Detalle: {motivo_regla}"
                else:
                    mensaje_horario = base
                en_infraccion = True
            else:
                base = (
                    "En este momento estás en horario de pico y placa. "
                    "Si el vehículo circula, estaría violando la normativa de pico y placa."
                )
                if motivo_regla:
                    mensaje_horario = f"{base} Detalle: {motivo_regla}"
                else:
                    mensaje_horario = base
                en_infraccion = False

    if tiene_pico_hoy:
        motivo_pico = (
            f"Hoy la placa {plate_up} TIENE pico y placa "
            f"(último dígito {ultimo}). No puede circular entre "
            f"{HORARIO_INICIO.strftime('%H:%M')} y {HORARIO_FIN.strftime('%H:%M')}."
        )
    else:
        if es_fin_de_semana or es_festivo:
            motivo_pico = "Hoy no aplica pico y placa porque es fin de semana o festivo."
        else:
            motivo_pico = (
                f"Hoy la placa {plate_up} NO tiene pico y placa "
                f"(último dígito {ultimo})."
            )

    st_enriched = {
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
    }
    return st_enriched


@app.get("/placas")
def listar_placas():
    """
    Endpoint técnico. No se expone en las interfaces de usuario del proyecto.
    """
    with get_conn() as c:
        cur = c.execute(
            "SELECT plate, color, first_seen, last_seen FROM vehicles "
            "ORDER BY last_seen DESC"
        )
        rows = cur.fetchall()

    placas = []
    for plate, color, first_seen, last_seen in rows:
        placas.append(
            {
                "plate": plate,
                "color": color or "DESCONOCIDO",
                "first_seen": first_seen,
                "last_seen": last_seen,
            }
        )

    return {"total": len(placas), "results": placas}


@app.get("/preguntar")
def preguntar(
    text: str = Query(..., description="Ej: Cualquier pregunta sobre placas o infracciones")
):
    """
    Endpoint principal del chatbot que usa IA + base de datos.
    """
    return responder_chat(text)


@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_ui():
    # Leer el HTML desde un string limpio
    html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Pico y Placa - Pasto</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: radial-gradient(circle at top, #1a2b4a 0%, #020617 45%, #000 100%);
            color: #e5e7eb;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            padding: 1.2rem 2rem;
            background: linear-gradient(90deg, #020617, #02101f);
            border-bottom: 1px solid #1f2937;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        }
        header h1 {
            font-size: 1.4rem;
            margin: 0;
        }
        header p {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: 0.25rem;
        }

        .chat-container {
            flex: 1;
            display: flex;
            justify-content: center;
            padding: 1.5rem;
            overflow: hidden;
        }
        .chat-card {
            position: relative;            /* 👈 necesario para posicionar el robot */
            width: 100%;
            max-width: 980px;
            background: radial-gradient(circle at top left, #0b1b3a 0%, #020617 55%, #020617 100%);
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            display: flex;
            flex-direction: column;
            overflow: visible;             /* 👈 permitir que el robot sobresalga */
            box-shadow:
                0 20px 60px rgba(0,0,0,0.65),
                0 0 0 1px rgba(15,23,42,0.8);
        }
        .toolbar {
            display: flex;
            justify-content: flex-end;
            gap: 0.5rem;
            padding: 0.7rem 1.2rem;
            border-bottom: 1px solid #1f2937;
            background: linear-gradient(90deg, #020617, #020a18);
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
        .chat-messages {
            flex: 1;
            padding: 1rem 1.3rem 1.1rem;
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
            background: linear-gradient(135deg, #1d4ed8, #22c55e);
            color: white;
            border-bottom-right-radius: 2px;
            box-shadow: 0 6px 18px rgba(37,99,235,0.6);
        }
        .msg.bot {
            margin-right: auto;
            background: rgba(15,23,42,0.96);
            border-bottom-left-radius: 2px;
            border: 1px solid rgba(30, 64, 175, 0.55);
            box-shadow: 0 10px 25px rgba(15,23,42,0.7);
        }

        /* 👇 Robot “saliendo” del cuadro, por fuera del recuadro */
        .robot-helper {
            position: absolute;
            right: -40px;          /* 👈 lo saca fuera del borde derecho */
            bottom: 4.2rem;        /* altura respecto al input */
            width: 130px;
            pointer-events: none;  /* no bloquea clics en el botón */
            filter: drop-shadow(0 14px 28px rgba(0,0,0,0.65));
        }

        .input-area {
            padding: 0.75rem 1.2rem 1.2rem;
            border-top: 1px solid #1f2937;
            background: linear-gradient(180deg, #020617, #000814);
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .input-area input {
            flex: 1;
            padding: 0.7rem 0.9rem;
            border-radius: 999px;
            border: 1px solid #4b5563;
            outline: none;
            background: rgba(15,23,42,0.95);
            color: #e5e7eb;
            font-size: 0.9rem;
        }
        .input-area input::placeholder {
            color: #6b7280;
        }
        .input-area input:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px rgba(37,99,235,0.5);
        }
        .input-area button {
            padding: 0.65rem 1.4rem;
            border-radius: 999px;
            border: none;
            background: radial-gradient(circle at 30% 0, #a7f3d0 0%, #22c55e 30%, #15803d 100%);
            color: #022c22;
            font-weight: 700;
            cursor: pointer;
            font-size: 0.9rem;
            box-shadow:
                0 0 0 1px rgba(22,163,74,0.6),
                0 14px 32px rgba(34,197,94,0.8);
        }
        .input-area button:disabled {
            opacity: 0.5;
            cursor: default;
            box-shadow: none;
        }

        @media (max-width: 768px) {
            header {
                padding: 1rem;
            }
            .chat-container {
                padding: 0.75rem;
            }
            .chat-card {
                border-radius: 14px;
            }
            .input-area {
                flex-direction: column;
                align-items: stretch;
            }
            .input-area button {
                width: 100%;
                justify-content: center;
            }
            .robot-helper {
                width: 100px;
                right: -20px;   /* también por fuera en móvil pero un poco menos */
                bottom: 5rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Chatbot Pico y Placa - Pasto</h1>
        <p>Pregunta en lenguaje natural sobre tu placa, días de restricción, posibles infracciones o el horario de pico y placa.</p>
    </header>

    <div class="chat-container">
        <div class="chat-card">
            <div class="toolbar">
                <button type="button" id="btn_clear">🧹 Limpiar chat</button>
                <button type="button" id="btn_ejemplo1"></button>
                <button type="button" id="btn_ejemplo2"></button>
            </div>

            <div id="messages" class="chat-messages"></div>

            <!-- 🤖 Robot asomado, justo encima del área de entrada -->
            <img src="/static/robot.png"
                 alt="Asistente de tránsito"
                 class="robot-helper" />

            <div class="input-area">
                <input type="text" id="user_input" placeholder="Escribe tu pregunta aquí...">
                <button type="button" id="send_btn">➤ Enviar</button>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        (function() {
            console.log("Script iniciando...");
            
            var API_URL = window.location.origin;
            var messagesDiv = document.getElementById("messages");
            var inputEl = document.getElementById("user_input");
            var sendBtn = document.getElementById("send_btn");
            var btnClear = document.getElementById("btn_clear");
            var btnEjemplo1 = document.getElementById("btn_ejemplo1");
            var btnEjemplo2 = document.getElementById("btn_ejemplo2");

            function addMessage(text, sender, meta, jsonDetail) {
                sender = sender || "bot";
                meta = meta || "";
                jsonDetail = jsonDetail || null;

                var msg = document.createElement("div");
                msg.className = "msg " + sender;
                msg.textContent = text;

                var wrapper = document.createElement("div");
                wrapper.appendChild(msg);

                if (meta) {
                    var metaDiv = document.createElement("div");
                    metaDiv.className = "meta";
                    metaDiv.textContent = meta;
                    wrapper.appendChild(metaDiv);
                }

                messagesDiv.appendChild(wrapper);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function sendMessage() {
                var text = inputEl.value.trim();
                if (!text) {
                    return;
                }

                addMessage(text, "user", "", null);
                inputEl.value = "";
                sendBtn.disabled = true;

                var url = API_URL + "/preguntar?text=" + encodeURIComponent(text);

                fetch(url)
                    .then(function(response) {
                        if (!response.ok) {
                            throw new Error("Error HTTP: " + response.status);
                        }
                        return response.json();
                    })
                    .then(function(data) {
                        console.log("Respuesta API:", data);
                        var respuesta = data.respuesta || "[Sin respuesta]";
                        addMessage(respuesta, "bot", "", null); // sin tipo ni JSON
                    })
                    .catch(function(error) {
                        console.error("Error:", error);
                        addMessage(
                            "Ocurrió un error al contactar con el servidor.",
                            "bot",
                            "",
                            null
                        );
                    })
                    .finally(function() {
                        sendBtn.disabled = false;
                    });
            }

            function clearChat() {
                messagesDiv.innerHTML = "";
                addMessage(
                    "Bienvenido soy tu asistente de pico y placa.",
                    "bot",
                    "",
                    null
                );
            }

            function sendExample(text) {
                inputEl.value = text;
                sendMessage();
            }

            sendBtn.addEventListener("click", function(e) {
                e.preventDefault();
                sendMessage();
            });

            inputEl.addEventListener("keypress", function(e) {
                if (e.key === "Enter") {
                    e.preventDefault();
                    sendMessage();
                }
            });

            btnClear.addEventListener("click", function(e) {
                e.preventDefault();
                clearChat();
            });

            btnEjemplo1.addEventListener("click", function(e) {
                e.preventDefault();
                sendExample("Qué dígitos tienen pico y placa hoy?");
            });

            btnEjemplo2.addEventListener("click", function(e) {
                e.preventDefault();
                sendExample("Si mi placa es ABC123, hoy puedo circular?");
            });

            window.addEventListener("load", function() {
                clearChat();
            });
            
            clearChat();
        })();
    </script>
</body>
</html>"""
    
    return HTMLResponse(content=html_content)