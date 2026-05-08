import os
import json
import copy
import re
from typing import List, Optional, Any, Dict, Tuple, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.qdrant_client_setup import get_qdrant_client

ANALYSIS_PROMPT = """Eres un Analista Experto en Negocios y Estrategia.
Te han presentado una nueva iniciativa con un nivel de detalle profundo:

- Título: {titulo}
- Unidad de Negocio: {unidad}

OBJETIVO ESTRATÉGICO:
- Problema/Oportunidad: {problema_oportunidad}
- Resultado Esperado: {resultado_esperado}
- Definición de MVP: {mvp}

DATOS E IMPACTO:
- Datos Necesarios: {datos_necesarios} (Dónde están: {datos_ubicacion})
- Impacto Operativo: {impacto_operacion}

VALIDACIÓN Y KPIS:
- Metodología de Validación: {validacion_exito}
- KPIs Propuestos: 
{kpis_str}

VALOR DEL NEGOCIO:
- Beneficio Cualitativo: {beneficio_esperado}
- Valor Estimado: {valor_estimado}

{strategic_context}

Por favor, realiza un análisis crítico, objetivo y estructurado de esta iniciativa:
1. Evaluación de Viabilidad y Congruencia: ¿Es el MVP realista considerando el Impacto Operativo y los Datos disponibles?
2. Puntos Fuertes: ¿Qué aspecto del planteamiento es robusto?
3. Puntos Ciegos / Riesgos: Observando los KPIs y la validación, ¿qué se está omitiendo o qué podría fallar?
4. Recomendación Estratégica: ¿Qué sugieres pivotar, medir o ajustar antes de proceder?
5. Alineación al Plan Estratégico: Contrasta la iniciativa contra el contexto estratégico recuperado desde Qdrant. Indica:
   - Nivel de alineación: Alta, Media, Baja o Sin evidencia suficiente.
   - Ejes/objetivos estratégicos relacionados.
   - Evidencia concreta tomada del contexto, citando la fuente si aparece.
   - Ajustes recomendados para mejorar la alineación.

Presenta tus resultados en formato Markdown, de forma analítica y colaborativa.
Reglas de formato para legibilidad:
- No uses encabezados H1 (`#`).
- Inicia con `## Análisis crítico de la iniciativa`.
- En la siguiente línea coloca `**Iniciativa:** {titulo}`.
- Usa encabezados `###` para las secciones principales.
- Mantén títulos cortos y separa los párrafos con saltos de línea.
"""

# Tool schemas for structured updates
class UpdateFormField(BaseModel):
    """Actualiza un campo de texto simple en el formulario de la iniciativa."""
    field_name: str = Field(..., description="El nombre técnico del campo (titulo, unidad, problema_oportunidad, resultado_esperado, mvp, datos_necesarios, datos_ubicacion, impacto_operacion, validacion_exito, beneficio_esperado, valor_estimado)")
    value: str = Field(..., description="El nuevo valor para el campo en texto claro. En 'valor_estimado' puedes incluir monto y contexto de cálculo/supuestos.")

class KpiItem(BaseModel):
    """Un KPI del formulario: nombre del indicador, línea base y meta."""
    indicador: str = Field(..., description="Nombre del indicador o métrica (ej. Retención mensual, NPS).")
    base: str = Field(..., description="Valor o situación actual (línea base).")
    meta: str = Field(..., description="Meta o valor deseado a alcanzar.")

class UpdateKpis(BaseModel):
    """Reemplaza la lista completa de KPIs. Incluye todos los filas: las existentes más las nuevas o editadas (merge manual). Cada item debe tener indicador, base y meta (strings, pueden estar vacíos solo si es fila en blanco)."""
    items: List[KpiItem] = Field(
        ...,
        description="Array de KPIs. Siempre envía el listado entero con la forma [{indicador, base, meta}, ...]. Mínimo un elemento.",
    )

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def get_strategic_context(query: str, limit: int = 5) -> str:
    """Busca fragmentos relevantes en la base de conocimientos estratégica."""
    client = get_qdrant_client()
    if not client:
        return ""
    
    try:
        embeddings = OpenAIEmbeddings()
        vector = embeddings.embed_query(query)
        
        results = client.search(
            collection_name="strategic_docs",
            query_vector=vector,
            limit=limit
        )
        
        context = "\nCONTEXTO ESTRATÉGICO Y NORMATIVO:\n"
        for res in results:
            source = res.payload.get("source", "Desconocido")
            text = res.payload.get("text", "")
            context += f"--- Fuente: {source} ---\n{text}\n\n"
        return context
    except Exception as e:
        print(f"Error searching RAG: {e}")
        return ""


def _normalize_analysis_markdown(text: str, title: Optional[str]) -> str:
    """Evita H1 enormes y separa títulos largos para mejorar lectura en el front."""
    body = (text or "").strip()
    if not body:
        return body

    clean_title = (title or "Sin título").strip() or "Sin título"
    body = re.sub(r"(?m)^#\s+", "## ", body)

    header_pattern = r"(?im)^\s*##\s+Análisis crítico de la iniciativa:?.*$"
    initiative_pattern = r"(?im)^\s*(?:\*\*)?Iniciativa:(?:\*\*)?\s*(.+?)\s*$"

    initiative_match = re.search(initiative_pattern, body[:500])
    initiative_title = (
        initiative_match.group(1).strip().strip("*") if initiative_match else clean_title
    )

    body = re.sub(header_pattern, "", body, count=1).lstrip()
    body = re.sub(initiative_pattern, "", body).lstrip()

    return (
        "## Análisis crítico de la iniciativa\n\n"
        f"**Iniciativa:** {initiative_title}\n\n"
        f"{body}"
    ).strip()

def analyze_initiative(data: dict) -> str:
    llm = get_llm()
    
    kpis_str = ""
    for k in data.get('kpis', []):
        kpis_str += f"- {k.get('indicador')} (Base: {k.get('base')}, Meta: {k.get('meta')})\n"

    strategic_query = "\n".join(
        str(data.get(key) or "")
        for key in (
            "titulo",
            "unidad",
            "problema_oportunidad",
            "resultado_esperado",
            "mvp",
            "impacto_operacion",
            "beneficio_esperado",
        )
    ).strip()
    strategic_context = get_strategic_context(strategic_query, limit=7)
    if not strategic_context:
        strategic_context = (
            "CONTEXTO ESTRATÉGICO Y NORMATIVO:\n"
            "No se recuperó contexto desde Qdrant. En la sección de alineación, "
            "indica que no hay evidencia suficiente para validar alineación contra el plan estratégico.\n"
        )
        
    prompt_template = PromptTemplate(
        input_variables=[
            "titulo", "unidad", "problema_oportunidad", "resultado_esperado", 
            "mvp", "datos_necesarios", "datos_ubicacion", 
            "impacto_operacion", "validacion_exito", 
            "beneficio_esperado", "valor_estimado",
            "kpis_str", "strategic_context"
        ],
        template=ANALYSIS_PROMPT
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({
        "titulo": data.get("titulo"),
        "unidad": data.get("unidad"),
        "problema_oportunidad": data.get("problema_oportunidad"),
        "resultado_esperado": data.get("resultado_esperado"),
        "mvp": data.get("mvp"),
        "datos_necesarios": data.get("datos_necesarios"),
        "datos_ubicacion": data.get("datos_ubicacion"),
        "impacto_operacion": data.get("impacto_operacion"),
        "validacion_exito": data.get("validacion_exito"),
        "beneficio_esperado": data.get("beneficio_esperado"),
        "valor_estimado": data.get("valor_estimado"),
        "kpis_str": kpis_str,
        "strategic_context": strategic_context,
    })
    return _normalize_analysis_markdown(result["text"], data.get("titulo"))

def _normalize_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Compatibilidad entre versiones de LangChain / formatos de tool_calls (dict u objetos)."""
    raw = getattr(response, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name")
            args = tc.get("args")
            if not name and "function" in tc:
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name")
                    a = fn.get("arguments")
                    if isinstance(a, str):
                        try:
                            args = json.loads(a) if a else {}
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = a or {}
        else:
            name = getattr(tc, "name", None)
            args = getattr(tc, "args", None) or {}
        if not name:
            continue
        if not isinstance(args, dict):
            args = {}
        out.append({"name": str(name), "args": args})
    return out

_GUIDED_SCALAR_ORDER = [
    "problema_oportunidad",
    "resultado_esperado",
    "titulo",
    "unidad",
    "mvp",
    "datos_necesarios",
    "datos_ubicacion",
    "impacto_operacion",
    "validacion_exito",
    "beneficio_esperado",
    "valor_estimado",
]

_GENERIC_ASSISTANT_REPLIES = {
    "he actualizado el formulario con la información proporcionada.",
    "he actualizado el formulario con la información proporcionada",
    "he actualizado el formulario.",
    "he actualizado el formulario",
}

_GUIDED_START_MESSAGES = {
    "",
    "hola",
    "inicio",
    "iniciar",
    "comenzar",
    "empezar",
    "modo guiado",
    "iniciar modo guiado",
    "start guided mode",
    "guided mode",
}

_GUIDED_IDEA_PROMPT = (
    "Para comenzar, cuéntame la idea en tus palabras: "
    "¿qué problema u oportunidad quieres resolver?"
)

_GENERIC_USER_CONFIRMATION_PHRASES = {
    "si",
    "sí",
    "ok",
    "oki",
    "dale",
    "de acuerdo",
    "correcto",
    "claro",
    "perfecto",
    "continuemos",
    "adelante",
    "esta bien",
    "está bien",
    "estan bien",
    "están bien",
    "asi esta bien",
    "así está bien",
    "si esta bien",
    "sí está bien",
    "si asi me parece",
    "si así me parece",
    "me parece bien",
    "me parece excelente",
    "me gusta",
    "lo dejamos así",
    "lo dejamos asi",
    "dejemoslo asi",
    "dejémoslo así",
    "sigamos",
    "continua",
    "continúa",
    "listo",
}


def _normalize_text_for_match(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("¡", "").replace("!", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = (
        normalized.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    return normalized


def _is_empty_field(key: str, value: Any) -> bool:
    if value is None:
        return True
    if key == "valor_estimado":
        if isinstance(value, (int, float)):
            return False
        s = str(value).strip()
        if not s:
            return True
        return False
    if isinstance(value, (int, float)):
        return False
    return str(value).strip() == ""


def _is_blank_guided_form(current_form: Optional[dict]) -> bool:
    if not current_form:
        return True
    merged = _apply_form_updates(current_form, [])
    return _next_guided_target(merged) == ("scalar", "problema_oportunidad")


def _is_guided_start_message(message: str) -> bool:
    normalized = (message or "").strip().lower()
    if normalized in _GUIDED_START_MESSAGES:
        return True
    frontend_start_markers = (
        "iniciemos en modo guiado",
        "iniciar modo guiado",
        "primera pregunta",
        "título de la iniciativa",
        "titulo de la iniciativa",
    )
    return any(marker in normalized for marker in frontend_start_markers)


def _looks_like_guided_idea_prompt(content: str) -> bool:
    normalized = (content or "").strip().lower()
    return "cuéntame la idea" in normalized and "problema u oportunidad" in normalized


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _is_affirmative_message(message: str) -> bool:
    normalized = (message or "").strip().lower()
    if not normalized:
        return False
    normalized_simple = _normalize_text_for_match(normalized)
    if normalized in _GENERIC_USER_CONFIRMATION_PHRASES or normalized_simple in _GENERIC_USER_CONFIRMATION_PHRASES:
        return True
    fragments = (
        "me parece bien",
        "me parece excelente",
        "me gusta",
        "lo dejamos así",
        "lo dejamos asi",
        "sigamos",
        "continua",
        "continúa",
        "esta bien",
        "está bien",
        "estan bien",
        "están bien",
        "asi me parece",
        "así me parece",
    )
    if any(frag in normalized for frag in fragments):
        return True
    tokens = set(normalized_simple.split())
    if "dejemoslo" in tokens and "asi" in tokens:
        return True
    if "es" in tokens and ("correcta" in tokens or "correcto" in tokens):
        return True
    if "bien" in tokens and ("si" in tokens or "asi" in tokens):
        return True
    return False


def _is_generic_update_reply(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    norm = _normalize_text_for_match(raw)
    return (
        "he actualizado el formulario" in norm
        or "formulario actualizado" in norm
        or norm in _GENERIC_ASSISTANT_REPLIES
    )


def _extract_text_between_quotes(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    quote_pairs = [("“", "”"), ('"', '"'), ("'", "'")]
    for left, right in quote_pairs:
        start = raw.find(left)
        if start == -1:
            continue
        end = raw.find(right, start + 1)
        if end == -1:
            continue
        candidate = raw[start + 1:end].strip()
        if candidate:
            return candidate
    return ""


def _get_last_assistant_message(history: List[dict]) -> str:
    for msg in reversed(history or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("assistant", "ai"):
            return str(msg.get("content") or "")
    return ""


def _proposal_from_last_assistant(
    history: List[dict],
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> Optional[str]:
    if not target or target[0] != "scalar":
        return None
    key = target[1]
    content = _get_last_assistant_message(history).strip()
    if not content:
        return None
    low = content.lower()
    checks = {
        "resultado_esperado": ("resultado esperado",),
        "titulo": ("título", "titulo"),
        "unidad": ("unidad de negocio",),
        "mvp": ("mvp", "primera versión", "primera version"),
        "datos_necesarios": ("datos necesarios",),
        "datos_ubicacion": ("ubicación de datos", "ubicacion de datos", "dónde están los datos", "donde estan los datos"),
        "impacto_operacion": ("impacto operativo",),
        "validacion_exito": ("validación del éxito", "validacion del exito", "cómo validar", "como validar"),
        "beneficio_esperado": ("beneficio cualitativo",),
        "valor_estimado": ("valor estimado",),
    }
    markers = checks.get(key, ())
    if markers and not any(m in low for m in markers) and key not in low:
        return None
    extracted = _extract_text_between_quotes(content)
    if extracted:
        return extracted
    if key == "valor_estimado":
        numeric = re.search(r"(\d+(?:\.\d+)?)", low)
        if numeric:
            return numeric.group(1)
        return content
    return None


def _suggest_title_from_form(merged: dict) -> str:
    current = str(merged.get("titulo") or "").strip()
    if current:
        return current
    problem = str(merged.get("problema_oportunidad") or "").strip()
    result = str(merged.get("resultado_esperado") or "").strip()
    text = f"{problem} {result}".lower()
    if any(k in text for k in ("qdrant", "rag", "llm", "modelo de lenguaje")):
        return "Mejora del LLM con RAG y Qdrant"
    if any(k in text for k in ("cliente", "atención", "soporte")):
        return "Optimización de Respuestas al Cliente con IA"
    if any(k in text for k in ("ventas", "comercial", "cotiz")):
        return "Asistente IA para Priorización Comercial"
    return "Optimización Operativa con Analítica e IA"


def _suggest_unit_from_form(merged: dict) -> str:
    current = str(merged.get("unidad") or "").strip()
    if current:
        return current
    blob = " ".join(
        [
            str(merged.get("problema_oportunidad") or ""),
            str(merged.get("resultado_esperado") or ""),
            str(merged.get("datos_necesarios") or ""),
            str(merged.get("impacto_operacion") or ""),
        ]
    ).lower()
    if any(k in blob for k in ("cliente", "atención", "soporte", "pqrs")):
        return "Servicio al Cliente"
    if any(k in blob for k in ("ventas", "comercial", "cotiz", "prospect")):
        return "Comercial y Ventas"
    if any(k in blob for k in ("finanza", "cost", "presupuesto", "gasto")):
        return "Finanzas"
    if any(k in blob for k in ("operación", "operacion", "proceso", "logística", "logistica")):
        return "Operaciones"
    if any(k in blob for k in ("talento", "rrhh", "reclutamiento", "personas")):
        return "Talento Humano"
    return "Tecnología e Innovación"


def _suggest_result_from_form(merged: dict) -> str:
    current = str(merged.get("resultado_esperado") or "").strip()
    if current:
        return current
    blob = " ".join(
        [
            str(merged.get("problema_oportunidad") or ""),
            str(merged.get("titulo") or ""),
        ]
    ).lower()
    if any(k in blob for k in ("qdrant", "rag", "llm", "modelo de lenguaje")):
        return "Reducir errores de interpretación y mejorar la calidad de las respuestas del modelo con información contextual confiable."
    return "Reducir tiempos y errores del proceso actual, con resultados más consistentes y medibles en la operación."


def _suggest_mvp_from_form(merged: dict) -> str:
    current = str(merged.get("mvp") or "").strip()
    if current:
        return current
    blob = " ".join(
        [str(merged.get("problema_oportunidad") or ""), str(merged.get("resultado_esperado") or "")]
    ).lower()
    if any(k in blob for k in ("qdrant", "rag", "llm", "modelo de lenguaje")):
        return "Piloto de 4 semanas integrando una base de conocimiento en Qdrant para responder solo un caso de uso crítico con medición de precisión y tiempo de respuesta."
    return "Piloto corto enfocado en un proceso crítico, con alcance limitado y métricas claras para validar impacto antes de escalar."


def _suggest_data_needed_from_form(merged: dict) -> str:
    current = str(merged.get("datos_necesarios") or "").strip()
    if current:
        return current
    blob = " ".join(
        [str(merged.get("problema_oportunidad") or ""), str(merged.get("mvp") or "")]
    ).lower()
    if any(k in blob for k in ("qdrant", "rag", "llm", "modelo de lenguaje")):
        return "Histórico de preguntas frecuentes, documentos normativos, respuestas validadas por expertos y trazas de consultas para evaluar calidad."
    return "Registros del proceso actual, tiempos de ciclo, incidencias, reprocesos y datos de salida para comparar antes y después."


def _suggest_data_location_from_form(merged: dict) -> str:
    current = str(merged.get("datos_ubicacion") or "").strip()
    if current:
        return current
    blob = " ".join(
        [
            str(merged.get("problema_oportunidad") or ""),
            str(merged.get("resultado_esperado") or ""),
            str(merged.get("mvp") or ""),
            str(merged.get("datos_necesarios") or ""),
        ]
    ).lower()
    if any(k in blob for k in ("sql", "mariadb", "mysql", "postgres", "postgresql", "oracle", "db sql")):
        return "Base de datos relacional SQL (por ejemplo MariaDB) del sistema transaccional."
    return "Repositorio documental interno (SharePoint/Drive), base transaccional del sistema operativo y reportes históricos en Excel."


def _suggest_operational_impact_from_form(merged: dict) -> str:
    current = str(merged.get("impacto_operacion") or "").strip()
    if current:
        return current
    return "El equipo reducirá tiempo de búsqueda y revisión manual, con menos reprocesos y mayor consistencia en las respuestas entregadas."


def _suggest_validation_from_form(merged: dict) -> str:
    current = str(merged.get("validacion_exito") or "").strip()
    if current:
        return current
    return "Medir durante 4 semanas el antes/después en tiempo de respuesta, tasa de errores y satisfacción del usuario interno."


def _suggest_qualitative_benefit_from_form(merged: dict) -> str:
    current = str(merged.get("beneficio_esperado") or "").strip()
    if current:
        return current
    return "Mejor calidad de servicio, mayor confianza en la información y decisiones más rápidas basadas en evidencia."


def _suggest_value_from_form(merged: dict) -> str:
    current = str(merged.get("valor_estimado") or "").strip()
    if current:
        return current
    return "USD 12,000 anuales estimados, considerando ahorro de tiempo operativo y reducción de reprocesos."


def _guided_specific_proposal(
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]], merged: dict
) -> Optional[str]:
    if not target or target[0] != "scalar":
        return None
    key = target[1]
    if key == "titulo":
        title = _suggest_title_from_form(merged)
        return f"Te propongo este título para la iniciativa: “{title}”. ¿Lo dejamos así?"
    if key == "unidad":
        unit = _suggest_unit_from_form(merged)
        return f"Te propongo esta unidad de negocio: “{unit}”. ¿Es correcta o la ajustamos?"
    if key == "resultado_esperado":
        result = _suggest_result_from_form(merged)
        return f"Te propongo este resultado esperado: “{result}”. ¿Lo dejamos así?"
    if key == "mvp":
        mvp = _suggest_mvp_from_form(merged)
        return f"Te propongo este MVP: “{mvp}”. ¿Lo dejamos así?"
    if key == "datos_necesarios":
        data = _suggest_data_needed_from_form(merged)
        return f"Te propongo estos datos necesarios: “{data}”. ¿Lo dejamos así?"
    if key == "datos_ubicacion":
        loc = _suggest_data_location_from_form(merged)
        return f"Te propongo esta ubicación de datos: “{loc}”. ¿Coincide con tu operación?"
    if key == "impacto_operacion":
        impact = _suggest_operational_impact_from_form(merged)
        return f"Te propongo este impacto en la operación: “{impact}”. ¿Lo dejamos así?"
    if key == "validacion_exito":
        val = _suggest_validation_from_form(merged)
        return f"Te propongo esta validación del éxito: “{val}”. ¿Te sirve?"
    if key == "beneficio_esperado":
        benefit = _suggest_qualitative_benefit_from_form(merged)
        return f"Te propongo este beneficio esperado: “{benefit}”. ¿Lo dejamos así?"
    if key == "valor_estimado":
        value = _suggest_value_from_form(merged)
        return (
            "Te propongo este valor estimado con contexto de negocio: "
            f"“{value}”. ¿Lo dejamos así o ajustamos supuestos?"
        )
    return None


def _default_value_for_scalar_target(
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]], merged: dict
) -> Optional[str]:
    if not target or target[0] != "scalar":
        return None
    key = target[1]
    if key == "resultado_esperado":
        return _suggest_result_from_form(merged)
    if key == "titulo":
        return _suggest_title_from_form(merged)
    if key == "unidad":
        return _suggest_unit_from_form(merged)
    if key == "mvp":
        return _suggest_mvp_from_form(merged)
    if key == "datos_necesarios":
        return _suggest_data_needed_from_form(merged)
    if key == "datos_ubicacion":
        return _suggest_data_location_from_form(merged)
    if key == "impacto_operacion":
        return _suggest_operational_impact_from_form(merged)
    if key == "validacion_exito":
        return _suggest_validation_from_form(merged)
    if key == "beneficio_esperado":
        return _suggest_qualitative_benefit_from_form(merged)
    if key == "valor_estimado":
        return _suggest_value_from_form(merged)
    return None


def _extract_user_value_for_target(field_name: str, message: str) -> Optional[str]:
    raw = (message or "").strip()
    if not raw:
        return None
    normalized_raw = _normalize_text_for_match(raw)
    if (
        _is_affirmative_message(raw)
        or normalized_raw in _GENERIC_USER_CONFIRMATION_PHRASES
        or len(normalized_raw) <= 18 and normalized_raw in _GENERIC_USER_CONFIRMATION_PHRASES
    ):
        return None
    if field_name == "valor_estimado":
        return raw
    if field_name == "datos_ubicacion":
        low = raw.lower()
        if any(k in low for k in ("sql", "mariadb", "mysql", "postgres", "postgresql", "oracle")):
            db = "SQL"
            if "mariadb" in low:
                db = "MariaDB"
            elif "mysql" in low:
                db = "MySQL"
            elif "postgresql" in low or "postgres" in low:
                db = "PostgreSQL"
            elif "oracle" in low:
                db = "Oracle"
            return f"Base de datos relacional en {db}."
    return raw


def _is_generic_confirmation_value(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    norm = _normalize_text_for_match(text)
    if norm in _GENERIC_USER_CONFIRMATION_PHRASES:
        return True
    tokens = set(norm.split())
    if "dejemoslo" in tokens and "asi" in tokens:
        return True
    if "es" in tokens and ("correcta" in tokens or "correcto" in tokens):
        return True
    if len(norm) <= 24 and ("ok" in tokens or "bien" in tokens or "perfecto" in tokens):
        return True
    return False


def _infer_datos_ubicacion_from_message(message: str) -> Optional[str]:
    low = (message or "").strip().lower()
    if not low:
        return None
    if any(k in low for k in ("mariadb", "mysql", "postgres", "postgresql", "oracle", "sql", "base de datos", "db sql")):
        if "mariadb" in low:
            return "Base de datos relacional en MariaDB."
        if "mysql" in low:
            return "Base de datos relacional en MySQL."
        if "postgresql" in low or "postgres" in low:
            return "Base de datos relacional en PostgreSQL."
        if "oracle" in low:
            return "Base de datos relacional en Oracle."
        return "Base de datos relacional SQL del sistema transaccional."
    return None


def _is_generic_guided_placeholder(body: str, target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]]) -> bool:
    if not body or not target or target[0] != "scalar":
        return False
    key = target[1]
    low = body.lower().strip()
    placeholder_markers = {
        "resultado_esperado": ("propón el resultado esperado",),
        "titulo": ("te propongo un título corto para la iniciativa", "te propongo un titulo corto para la iniciativa"),
        "unidad": ("te propongo la unidad de negocio más probable",),
        "mvp": ("te propongo un mvp inicial",),
        "datos_necesarios": ("te propongo los datos necesarios",),
        "datos_ubicacion": ("te propongo dónde podrían estar esos datos", "te propongo donde podrian estar esos datos"),
        "impacto_operacion": ("te propongo el impacto operativo esperado",),
        "validacion_exito": ("te propongo cómo validar el éxito", "te propongo como validar el exito"),
        "beneficio_esperado": ("te propongo el beneficio cualitativo",),
        "valor_estimado": ("si tienes una cifra, dime el valor estimado",),
    }
    markers = placeholder_markers.get(key, ())
    if any(m in low for m in markers):
        return True
    if low.startswith("te propongo ") and "¿" not in body and "?" not in body:
        return True
    return False


def _draft_initial_idea_fields(message: str) -> Dict[str, str]:
    """Reformula una idea inicial vaga en lenguaje de negocio para el formulario guiado."""
    idea = (message or "").strip()
    if not idea:
        return {"problema_oportunidad": "", "resultado_esperado": ""}
    try:
        llm = get_llm()
        response = llm.invoke(
            f"""
Eres consultor de negocio para usuarios no técnicos. Reformula la idea del usuario en lenguaje claro,
profesional y orientado a negocio. No copies literal. No uses jerga técnica innecesaria.

Devuelve SOLO JSON válido con estas llaves:
- problema_oportunidad: una redacción clara del problema u oportunidad para un formulario oficial.
- resultado_esperado: una propuesta concreta de qué debería mejorar si la iniciativa funciona.

Idea del usuario:
{idea}
"""
        )
        raw = getattr(response, "content", response)
        data = json.loads(_strip_json_fence(str(raw)))
        problem = str(data.get("problema_oportunidad") or "").strip()
        result = str(data.get("resultado_esperado") or "").strip()
        if problem:
            return {
                "problema_oportunidad": problem,
                "resultado_esperado": result,
            }
    except Exception:
        pass
    return {
        "problema_oportunidad": (
            "Existe una oportunidad de mejora relacionada con la idea planteada por el usuario: "
            f"{idea}"
        ),
        "resultado_esperado": "",
    }


def _apply_form_updates(
    current_form: Optional[dict], form_updates: List[Dict[str, Any]]
) -> dict:
    base = copy.deepcopy(current_form) if current_form else {}
    if "kpis" not in base or not isinstance(base.get("kpis"), list):
        base["kpis"] = [{"indicador": "", "base": "", "meta": ""}]
    for u in form_updates or []:
        name = (u.get("function") or u.get("name") or "").strip()
        args = u.get("args") or {}
        if name == "UpdateFormField":
            field_name = args.get("field_name")
            val = args.get("value")
            if field_name and field_name in _GUIDED_SCALAR_ORDER and field_name != "kpis":
                base[field_name] = val
        elif name == "UpdateKpis":
            raw_items = args.get("items") if args.get("items") is not None else args.get("kpis")
            if not raw_items:
                continue
            items: List[dict] = []
            for it in raw_items:
                if isinstance(it, dict):
                    items.append(
                        {
                            "indicador": str(it.get("indicador", "") or ""),
                            "base": str(it.get("base", "") or ""),
                            "meta": str(it.get("meta", "") or ""),
                        }
                    )
                else:
                    items.append({"indicador": "", "base": "", "meta": ""})
            if items:
                base["kpis"] = items
    return base


def _next_guided_target(merged: dict) -> Optional[Union[Tuple[str, str], Tuple[str, int, str]]]:
    for key in _GUIDED_SCALAR_ORDER:
        if _is_empty_field(key, merged.get(key)):
            return ("scalar", key)
    kpis = merged.get("kpis") or []
    if not kpis:
        return ("kpi", 0, "indicador")
    for i, row in enumerate(kpis):
        if not isinstance(row, dict):
            row = {}
        for sub in ("indicador", "base", "meta"):
            if _is_empty_field(sub, row.get(sub)):
                return ("kpi", i, sub)
    return None


def _guided_question(target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]]) -> str:
    if target is None:
        return (
            "Con esto tenemos un borrador completo. Revisa el formulario a la izquierda; "
            "si quieres, dime qué ajustar o si añadimos otro indicador (KPI)."
        )
    kind = target[0]
    if kind == "scalar":
        key = target[1]
        q = {
            "problema_oportunidad": "Cuéntame la idea en tus palabras: ¿qué problema u oportunidad quieres resolver?",
            "resultado_esperado": "Con base en la idea, propón el resultado esperado y dime si lo dejamos así.",
            "titulo": "Te propongo un título corto para la iniciativa; dime si lo dejamos o lo ajustamos.",
            "unidad": "Te propongo la unidad de negocio más probable; dime si es correcta.",
            "mvp": "Te propongo un MVP inicial y validamos si el alcance es correcto.",
            "datos_necesarios": "Te propongo los datos necesarios para ejecutarla; dime si faltan fuentes.",
            "datos_ubicacion": "Te propongo dónde podrían estar esos datos; dime si coincide con tu operación.",
            "impacto_operacion": "Te propongo el impacto operativo esperado; dime si lo ajustamos.",
            "validacion_exito": "Te propongo cómo validar el éxito; dime si esa medición te sirve.",
            "beneficio_esperado": "Te propongo el beneficio cualitativo; dime si refleja el valor real.",
            "valor_estimado": "Si tienes una cifra, dime el valor estimado; si no, te propongo un rango razonable para validar.",
        }
        return q.get(
            key,
            "Sigamos con el siguiente dato. ¿Puedes completar este campo en una frase?",
        )
    if kind != "kpi":
        return "¿Qué dato te gustaría ajustar a continuación?"
    _t, row_idx, sub = target
    n = row_idx + 1
    if sub == "indicador":
        return f"Te propongo el indicador n.º {n} para medir la iniciativa; dime si lo dejamos así."
    if sub == "base":
        return f"Te propongo una línea base para el KPI n.º {n}; dime si coincide con la situación actual."
    return f"Te propongo una meta para el KPI n.º {n}; dime si es alcanzable."


def _append_guided_followup(
    content: str, guided_mode: bool, current_form: Optional[dict], form_updates: List[Dict[str, Any]]
) -> str:
    if not guided_mode:
        return content
    merged = _apply_form_updates(current_form, form_updates)
    nxt = _next_guided_target(merged)
    line = _guided_question(nxt)
    forced_line = _guided_specific_proposal(nxt, merged)
    if forced_line:
        line = forced_line
    body = (content or "").strip()
    low = body.lower()
    if _is_generic_guided_placeholder(body, nxt):
        body = ""
        low = ""
    if not body:
        return line
    if _is_generic_update_reply(body):
        if nxt is None:
            return _guided_question(None)
        return f"Listo, avanzamos. {line}"
    if "?" in body and len(body) > 40:
        return body
    if line in body or line.rstrip(".") in body:
        return body
    if body.endswith("?") and nxt is not None and len(body) > 25:
        return body
    return f"{body}\n\n{line}" if body else line


def chat_with_agent(
    message: str,
    history: List[dict],
    current_form: Optional[dict] = None,
    guided_mode: bool = False,
) -> dict:
    if (
        guided_mode
        and _is_blank_guided_form(current_form)
        and _is_guided_start_message(message)
    ):
        return {"content": _GUIDED_IDEA_PROMPT, "form_updates": []}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35 if guided_mode else 0.5)
    from langchain.schema import SystemMessage, HumanMessage, AIMessage

    # Bind tools to LLM
    tools = [UpdateFormField, UpdateKpis]
    llm_with_tools = llm.bind_tools(tools)

    # 1. Obtener contexto estratégico (RAG)
    context = get_strategic_context(message)

    form_snapshot = ""
    if current_form is not None:
        try:
            # El modelo necesita el estado actual (sobre todo kpis) para fusionar o añadir filas
            snap = {k: v for k, v in current_form.items() if k in (
                "titulo", "unidad", "problema_oportunidad", "resultado_esperado", "mvp",
                "datos_necesarios", "datos_ubicacion", "impacto_operacion", "validacion_exito",
                "kpis", "beneficio_esperado", "valor_estimado",
            )}
            form_snapshot = "\n\nESTADO ACTUAL DEL FORMULARIO (JSON, úsalo para no perder datos al actualizar KPIs u otros campos):\n" + json.dumps(
                snap, ensure_ascii=False, indent=2
            )
        except Exception:
            form_snapshot = ""

    guided_extra = (
        """

MODO GUIADO (activo en esta petición):
- Tu misión es ayudar a usuarios no técnicos a transformar una idea vaga, problema operativo u oportunidad de mejora en una iniciativa de IA clara, completa y lista para registrar en el formulario oficial.
- El usuario puede no saber qué datos necesita, qué es un MVP, cómo validar la iniciativa o cómo definir KPIs. Actúa como consultor paciente, práctico y orientado a negocio.
- No hables como técnico. No asumas que el usuario conoce conceptos de IA. Usa preguntas simples, ejemplos concretos y propuestas editables.
- No pidas el título al inicio. Primero pide o interpreta la idea del usuario.
- A partir de la idea, trabaja como asesor: analiza, mejora y reformula en lenguaje de negocio antes de llenar campos. Nunca copies y pegues la idea cruda si puede escribirse mejor.
- Avanza en este orden: idea/problema, resultado esperado, título, unidad, MVP, datos necesarios, ubicación de datos, impacto, validación, beneficio cualitativo, valor estimado, luego KPIs (indicador → base → meta).
- Cada turno: como mucho UNA propuesta o UNA pregunta al usuario.
- Cada respuesta visible debe ser breve, amable, y terminar con una confirmación tipo "¿Lo dejamos así?" o una pregunta hacia el siguiente dato aún no guardado, salvo que el formulario esté completo.
- Tras cualquier 'UpdateFormField' o 'UpdateKpis', no te limites a decir que actualizaste: **continúa** con la pregunta siguiente.
- Si el usuario responde "sí", "correcto", "ok" o equivalente, toma como aprobada la propuesta anterior y guarda el campo correspondiente con la herramienta adecuada.
- Si el usuario no sabe, propone 1-2 opciones concretas. Puedes prellenar un borrador con tools cuando la idea dé evidencia suficiente, pero deja claro que es editable.
- Si propones un valor para un campo, menciona el nombre del campo en lenguaje natural para que el usuario sepa qué está aprobando.
- Antes de pedir cada campo, explica en una frase qué significa. Ejemplo: "Resultado esperado significa qué debería mejorar si esto funciona".
- Explica MVP así: "una primera versión simple para probar si la idea funciona, sin construir todo desde el inicio".
- Para datos necesarios, pregunta en lenguaje simple: qué información usa hoy el equipo, dónde vive (Excel, sistema, correos, PDFs, reportes) y quién la tiene.
- Explica validación como: "cómo sabremos que la iniciativa funcionó".
- Convierte beneficios vagos en KPIs sugeridos con indicador, línea base y meta.
- Si el usuario no sabe el valor estimado, ayuda a estimarlo con horas, personas involucradas, errores, reprocesos o demoras. Si no alcanza, propón dejarlo como estimación preliminar editable.
"""
    ) if guided_mode else ""

    messages = [
        SystemMessage(
            content=f"""Eres un Asesor Estratégico Experto y Consultor de Negocios para Familia Aldea. 
Tu objetivo es ayudar al usuario a completar su iniciativa, asegurando que esté alineada con el Plan Estratégico 2025-2029 y el Modelo de Entrada oficial.

{context}
{form_snapshot}

REGLAS DE ASESORÍA:
1. Actúa como asesor: Si el usuario dice algo que no se alinea con el contexto estratégico anterior, sugiérele mejoras de forma colaborativa.
2. Cita el plan: Si usas información de la base de conocimientos, menciona algo como "Según el plan estratégico..." o "El modelo de entrada sugiere...".
3. Relleno proactivo: Cada vez que el usuario proporcione información relevante para un campo de texto, usa 'UpdateFormField' con el field_name y value correctos. Reformula la respuesta en lenguaje profesional de negocio antes de guardarla.
3.1. Para 'valor_estimado', puedes devolver texto con monto + contexto (supuestos, periodo y fuente del cálculo). No lo limites a solo dígitos.
4. **KPIs (indicadores)**: No uses UpdateFormField para los KPIs. Usa la herramienta **'UpdateKpis'** con el array **'items'**: cada elemento debe incluir **indicador**, **base** y **meta** (strings). Si el usuario añade o modifica un KPI, devuelve **toda la lista** de KPIs (las filas anteriores del JSON de estado + las nuevas o corregidas), nunca un solo ítem suelto sin el resto.
5. Si el usuario menciona varios KPIs en un mensaje, consolídalos en un solo UpdateKpis.
6. Si estás en modo guiado y el formulario aún está vacío, trata el primer mensaje del usuario como la idea inicial; no le pidas un título antes de entender esa idea.

CAMPOS TÉCNICOS DISPONIBLES (field_name en UpdateFormField, exactamente así):
- 'titulo': Nombre de la iniciativa.
- 'unidad': Unidad de negocio.
- 'problema_oportunidad': El problema o la oportunidad.
- 'resultado_esperado': Qué se espera lograr.
- 'mvp': Producto Mínimo Viable.
- 'datos_necesarios': Qué datos se requieren.
- 'datos_ubicacion': Dónde están los datos.
- 'impacto_operacion': Qué cambiará en la operación.
- 'validacion_exito': Cómo se medirá el éxito.
- 'beneficio_esperado': Beneficio cualitativo.
- 'valor_estimado': Valor estimado con contexto (monto, periodo y supuestos clave).
{guided_extra}
"""
        )
    ]
    
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
            
    messages.append(HumanMessage(content=message))
    
    merged_before = _apply_form_updates(current_form, [])
    target_before = _next_guided_target(merged_before) if guided_mode else None
    response = llm_with_tools.invoke(messages)

    form_updates: List[Dict[str, Any]] = []
    for call in _normalize_tool_calls(response):
        form_updates.append(
            {
                "function": call["name"],
                "args": call.get("args") or {},
            }
        )

    # Sanitiza tool calls: evita guardar confirmaciones genéricas como valor de campos.
    for update in form_updates:
        if update.get("function") != "UpdateFormField":
            continue
        args = update.get("args") or {}
        field_name = args.get("field_name")
        if not field_name:
            continue
        value = args.get("value")
        if not _is_generic_confirmation_value(value):
            continue
        inferred = _proposal_from_last_assistant(history, ("scalar", str(field_name)))
        if not inferred:
            inferred = _default_value_for_scalar_target(("scalar", str(field_name)), merged_before)
        if inferred:
            args["value"] = inferred

    # Fallback defensivo: si el usuario confirma una propuesta en modo guiado y el LLM
    # no ejecutó tool call, inferimos el guardado del campo sugerido para no romper el flujo.
    if guided_mode and not form_updates and target_before and target_before[0] == "scalar":
        field_name = target_before[1]
        if _is_affirmative_message(message):
            inferred_value = _proposal_from_last_assistant(history, target_before) or _default_value_for_scalar_target(
                target_before, merged_before
            )
            if inferred_value:
                form_updates.append(
                    {
                        "function": "UpdateFormField",
                        "args": {
                            "field_name": field_name,
                            "value": inferred_value,
                        },
                    }
                )
        else:
            explicit_value = _extract_user_value_for_target(field_name, message)
            if explicit_value:
                form_updates.append(
                    {
                        "function": "UpdateFormField",
                        "args": {
                            "field_name": field_name,
                            "value": explicit_value,
                        },
                    }
                )
    # Si el usuario menciona explícitamente el origen SQL/BD, guarda también ubicación de datos
    # aunque el modelo no lo haya hecho en este turno.
    if guided_mode:
        has_location_update = any(
            u.get("function") == "UpdateFormField"
            and (u.get("args") or {}).get("field_name") == "datos_ubicacion"
            for u in form_updates
        )
        if not has_location_update and _is_empty_field("datos_ubicacion", merged_before.get("datos_ubicacion")):
            inferred_location = _infer_datos_ubicacion_from_message(message)
            if inferred_location:
                form_updates.append(
                    {
                        "function": "UpdateFormField",
                        "args": {
                            "field_name": "datos_ubicacion",
                            "value": inferred_location,
                        },
                    }
                )

    initial_draft: Optional[Dict[str, str]] = None
    if (
        guided_mode
        and _is_blank_guided_form(current_form)
        and not _is_guided_start_message(message)
        and (message or "").strip()
    ):
        problem_update = None
        for update in form_updates:
            args = update.get("args", {})
            if args.get("field_name") == "problema_oportunidad":
                problem_update = update
                break
        raw_problem = (
            str(problem_update.get("args", {}).get("value") or "").strip()
            if problem_update
            else ""
        )
        if not problem_update or raw_problem.lower() == message.strip().lower():
            initial_draft = _draft_initial_idea_fields(message)
            if problem_update:
                problem_update["args"]["value"] = initial_draft["problema_oportunidad"]
            else:
                form_updates.append(
                    {
                        "function": "UpdateFormField",
                        "args": {
                            "field_name": "problema_oportunidad",
                            "value": initial_draft["problema_oportunidad"],
                        },
                    }
                )

    if initial_draft:
        draft_result = initial_draft.get("resultado_esperado") or (
            "mejorar el proceso descrito, reduciendo errores, tiempos de revisión "
            "y respuestas inconsistentes."
        )

    content_out = _append_guided_followup(
        response.content or "", guided_mode, current_form, form_updates
    )
    if initial_draft:
        content_out = (
            "Te propongo redactar el problema u oportunidad así: "
            f"“{initial_draft['problema_oportunidad']}”\n\n"
            "Ahora definamos el resultado esperado. Esto significa qué debería mejorar "
            "si la iniciativa funciona. Te propongo: "
            f"“{initial_draft.get('resultado_esperado') or draft_result}”. ¿Lo dejamos así?"
        )
    if (
        guided_mode
        and _is_blank_guided_form(current_form)
        and not form_updates
        and "título" in (content_out or "").lower()
    ):
        content_out = _GUIDED_IDEA_PROMPT
    if (
        guided_mode
        and form_updates
        and any(
            update.get("args", {}).get("field_name") == "problema_oportunidad"
            for update in form_updates
        )
        and _looks_like_guided_idea_prompt(content_out)
    ):
        content_out = (
            "Perfecto, tomé esa idea como el problema u oportunidad inicial.\n\n"
            "Ahora definamos el resultado esperado. Esto significa qué debería mejorar "
            "si la iniciativa funciona. ¿Qué resultado concreto esperas lograr?"
        )
    if not (content_out or "").strip() and not guided_mode:
        content_out = (
            "He actualizado el formulario con la información proporcionada."
            if form_updates
            else ""
        )
    # Capa final de seguridad: evita respuestas genéricas cuando aún faltan campos/KPIs.
    if guided_mode and _is_generic_update_reply(content_out):
        merged_after = _apply_form_updates(current_form, form_updates)
        nxt_after = _next_guided_target(merged_after)
        if nxt_after is not None:
            forced = _guided_specific_proposal(nxt_after, merged_after) or _guided_question(nxt_after)
            content_out = f"Listo, avanzamos. {forced}"

    return {
        "content": content_out,
        "form_updates": form_updates,
    }

