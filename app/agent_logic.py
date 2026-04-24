import os
import json
import copy
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

Por favor, realiza un análisis crítico, objetivo y estructurado de esta iniciativa:
1. Evaluación de Viabilidad y Congruencia: ¿Es el MVP realista considerando el Impacto Operativo y los Datos disponibles?
2. Puntos Fuertes: ¿Qué aspecto del planteamiento es robusto?
3. Puntos Ciegos / Riesgos: Observando los KPIs y la validación, ¿qué se está omitiendo o qué podría fallar?
4. Recomendación Estratégica: ¿Qué sugieres pivotar, medir o ajustar antes de proceder?

Presenta tus resultados en formato Markdown, de forma analítica y colaborativa.
"""

# Tool schemas for structured updates
class UpdateFormField(BaseModel):
    """Actualiza un campo de texto simple en el formulario de la iniciativa."""
    field_name: str = Field(..., description="El nombre técnico del campo (titulo, unidad, problema_oportunidad, resultado_esperado, mvp, datos_necesarios, datos_ubicacion, impacto_operacion, validacion_exito, beneficio_esperado, valor_estimado)")
    value: str = Field(..., description="El nuevo valor para el campo. Para 'valor_estimado' usa solo números (ej. 5000 o 5000.75), sin $ ni comas.")

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

def analyze_initiative(data: dict) -> str:
    llm = get_llm()
    
    kpis_str = ""
    for k in data.get('kpis', []):
        kpis_str += f"- {k.get('indicador')} (Base: {k.get('base')}, Meta: {k.get('meta')})\n"
        
    prompt_template = PromptTemplate(
        input_variables=[
            "titulo", "unidad", "problema_oportunidad", "resultado_esperado", 
            "mvp", "datos_necesarios", "datos_ubicacion", 
            "impacto_operacion", "validacion_exito", 
            "beneficio_esperado", "valor_estimado",
            "kpis_str"
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
    })
    return result['text']

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
    "titulo",
    "unidad",
    "valor_estimado",
    "problema_oportunidad",
    "resultado_esperado",
    "mvp",
    "datos_necesarios",
    "datos_ubicacion",
    "impacto_operacion",
    "validacion_exito",
    "beneficio_esperado",
]

_GENERIC_ASSISTANT_REPLIES = {
    "he actualizado el formulario con la información proporcionada.",
    "he actualizado el formulario.",
}


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
            "titulo": "¿Cómo quieres llamar a la iniciativa? (un título claro y corto).",
            "unidad": "¿A qué unidad de negocio pertenece? (elige la que mejor aplique, por ejemplo: Aldea Global, Talento Humano, etc.).",
            "valor_estimado": "¿Cuál es el valor estimado en dólares? (solo el número, por ejemplo: 5000 o 15000.50).",
            "problema_oportunidad": "Describe el problema u oportunidad que quieres atacar.",
            "resultado_esperado": "¿Qué resultado concreto esperas lograr con esta iniciativa?",
            "mvp": "¿Cuál sería un MVP o alcance mínimo para la primera iteración?",
            "datos_necesarios": "¿Qué datos necesitas para ejecutarla? (fuentes, tablas, métricas).",
            "datos_ubicacion": "¿Dónde están o vivirán esos datos? (SAP, Excel, data lake, etc.).",
            "impacto_operacion": "¿Cómo cambiará el trabajo diario u operación del equipo?",
            "validacion_exito": "¿Cómo validarán que tuvo éxito? (criterio o medición).",
            "beneficio_esperado": "¿Qué beneficio cualitativo esperas (más allá del valor en $)?",
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
        return f"Definamos el indicador n.º {n}: ¿qué métrica o nombre de KPI quieres usar?"
    if sub == "base":
        return f"Para el KPI n.º {n}, ¿cuál es la línea base o situación actual?"
    return f"Para el KPI n.º {n}, ¿cuál es la meta o valor deseado?"


def _append_guided_followup(
    content: str, guided_mode: bool, current_form: Optional[dict], form_updates: List[Dict[str, Any]]
) -> str:
    if not guided_mode:
        return content
    merged = _apply_form_updates(current_form, form_updates)
    nxt = _next_guided_target(merged)
    line = _guided_question(nxt)
    body = (content or "").strip()
    low = body.lower()
    if not body:
        return line
    if low in _GENERIC_ASSISTANT_REPLIES:
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
- Avanza en orden: título, unidad, valor estimado, problema, resultado, MVP, datos, ubicación, impacto, validación, beneficio cualitativo, luego KPIs (indicador → base → meta).
- Cada turno: como mucho UNA pregunta al usuario.
- Cada respuesta tuya (texto visible) debe ser breve, amable, y **terminar con UNA pregunta** hacia el siguiente dato aún no guardado, salvo que el formulario esté completo.
- Tras cualquier 'UpdateFormField' o 'UpdateKpis', no te limites a decir que actualizaste: **continúa** con la pregunta siguiente.
- Si el usuario no sabe, propone 1-2 opciones, elige tú, llénalas vía tool y pasa a la **siguiente** pregunta.
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
3. Relleno proactivo: Cada vez que el usuario proporcione información relevante para un campo de texto, usa 'UpdateFormField' con el field_name y value correctos.
3.1. Para 'valor_estimado', devuelve siempre un valor numérico limpio (solo dígitos y opcional punto decimal), sin símbolos de moneda ni separadores de miles.
4. **KPIs (indicadores)**: No uses UpdateFormField para los KPIs. Usa la herramienta **'UpdateKpis'** con el array **'items'**: cada elemento debe incluir **indicador**, **base** y **meta** (strings). Si el usuario añade o modifica un KPI, devuelve **toda la lista** de KPIs (las filas anteriores del JSON de estado + las nuevas o corregidas), nunca un solo ítem suelto sin el resto.
5. Si el usuario menciona varios KPIs en un mensaje, consolídalos en un solo UpdateKpis.

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
- 'valor_estimado': Beneficio cuantitativo. IMPORTANTE: enviar solo número, sin '$' ni comas.
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
    
    response = llm_with_tools.invoke(messages)

    form_updates: List[Dict[str, Any]] = []
    for call in _normalize_tool_calls(response):
        form_updates.append(
            {
                "function": call["name"],
                "args": call.get("args") or {},
            }
        )

    content_out = _append_guided_followup(
        response.content or "", guided_mode, current_form, form_updates
    )
    if not (content_out or "").strip() and not guided_mode:
        content_out = (
            "He actualizado el formulario con la información proporcionada."
            if form_updates
            else ""
        )

    return {
        "content": content_out,
        "form_updates": form_updates,
    }

