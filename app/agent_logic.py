import os
import json
import copy
import re
import ast
from html.parser import HTMLParser
from typing import List, Optional, Any, Dict, Tuple, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.qdrant_client_setup import get_qdrant_client
from app.scoring import (
    BUSINESS_CRITERION_KEYS,
    apply_implementation_days_to_potenciadores,
    apply_time_to_value_from_days,
    build_potenciadores_payload,
    default_business_scores,
    normalize_business_scores,
    normalize_business_explanations,
)
from app.roi import (
    estimate_success_probability,
    money_field_prompt,
    money_field_text,
    normalize_beneficio_field,
    normalize_money_field,
    beneficio_field_text,
    effective_implementation_days,
)

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
- Días estimados de implementación: {dias_implementacion_estimados}

VALIDACIÓN Y KPIS:
- Metodología de Validación: {validacion_exito}
- KPIs Propuestos: 
{kpis_str}

VALOR DEL NEGOCIO:
{beneficio_esperado}
- Costo Estimado: {valor_estimado}

CALIDAD DE DATA Y DOCUMENTACIÓN:
{calidad_datos_context}

FACTOR INNOVADOR (evaluación previa):
{factor_innovador_context}

BIBLIOGRAFÍA Y CONFIABILIDAD:
{bibliografia_context}

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
6. Confiabilidad de fuentes: Resume si la bibliografía incluida es suficiente y confiable para sustentar la iniciativa.
7. Calidad de data y documentación: Evalúa si los datos declarados y la documentación respaldan la propuesta. Usa el contexto de calidad de data provisto.
8. Factor innovador: Evalúa si la iniciativa aporta novedad real frente a lo existente. Usa el contexto de factor innovador provisto.

Presenta tus resultados en formato Markdown, de forma analítica y colaborativa.
Reglas de formato para legibilidad:
- No uses encabezados H1 (`#`).
- Inicia con `## Análisis crítico de la iniciativa`.
- En la siguiente línea coloca `**Iniciativa:** {titulo}`.
- Usa encabezados `###` para las secciones principales.
- Mantén títulos cortos y separa los párrafos con saltos de línea.
"""

SCORE_SUGGESTION_PROMPT = """Eres un comité experto priorizando iniciativas de IA.
Con base en la información de la iniciativa y el análisis generado, asigna puntajes enteros de 1 a 5 para cada criterio.

Escala general:
1 = muy bajo o sin evidencia
2 = bajo
3 = medio
4 = alto
5 = muy alto

Criterios:
- financiero: impacto directo en ingresos, ahorro o reducción de pérdidas.
- estrategico: alineación con prioridades críticas de la organización.
- cliente: mejora de experiencia, acceso o tiempos del usuario final.
- datos_ia: generación o estructuración de datos y capacidades de analítica/IA reutilizables. Usa calidad_datos_analisis si está disponible.
- time_to_value: 5 si genera valor en 90 días o menos; si dias_implementacion_estimados > 90 el puntaje debe bajar (91-180 → máx 3; >180 → máx 2).

Complejidad y riesgo los evalúa TI por separado; no los incluyas en este score.

Devuelve solo JSON válido, sin markdown.
El JSON debe tener una clave "resumen" con una explicación general de una línea.
También debe tener una clave "criterios"; dentro incluye financiero, estrategico, cliente, datos_ia y time_to_value.
Cada criterio debe tener "puntaje" y "explicacion"; cada explicación debe ser corta, precisa y de una sola línea.

INICIATIVA:
{initiative_json}

ANÁLISIS:
{analysis}
"""

ROI_PROBABILITY_PROMPT = """Eres un comité experto estimando la probabilidad de éxito de iniciativas de IA.
Calcula una probabilidad entre 0 y 1 considerando:
- Madurez y disponibilidad de datos.
- Claridad del caso de negocio y KPIs.
- Alcance del MVP.
- Complejidad de ejecución y dependencias.
- Capacidad de adopción del negocio.

Devuelve solo JSON válido, sin markdown, con:
- p_exito: número entre 0 y 1.
- explicacion: una línea breve.

INICIATIVA:
{initiative_json}

ANÁLISIS:
{analysis}
"""

BIBLIOGRAPHY_RELIABILITY_PROMPT = """Eres un analista evaluando la confiabilidad de fuentes usadas para sustentar una iniciativa de negocio.
Evalúa cada URL con señales objetivas: reputación del dominio, claridad del contenido disponible, posible sesgo comercial, actualidad aparente y pertinencia para la iniciativa.

Devuelve solo JSON válido, sin markdown, con esta forma:
{
  "resumen": "síntesis breve",
  "riesgo_general": "bajo|medio|alto",
  "fuentes": [
    {
      "url": "...",
      "dominio": "...",
      "confiabilidad": "alta|media|baja",
      "estado": "contenido_obtenido|heuristica",
      "senales_favor": ["..."],
      "senales_alerta": ["..."],
      "recomendacion": "..."
    }
  ]
}

INICIATIVA:
{initiative_json}

FUENTES:
{sources_json}
"""

INITIATIVE_SUMMARY_PROMPT = """Resume esta iniciativa en 3 a 5 líneas para un comité ejecutivo.
Explica en síntesis: qué problema resuelve, cómo se implementaría, qué valor espera generar y qué fuentes la sustentan.
No uses markdown complejo ni listas largas.

INICIATIVA:
{initiative_json}

ANÁLISIS DE FUENTES:
{bibliografia_analisis}
"""

DATA_QUALITY_PROMPT = """Eres un analista de datos evaluando la calidad de la información y documentación que respalda una iniciativa.

Evalúa completitud, accesibilidad, trazabilidad documental y gobernanza de los datos declarados.

Devuelve solo JSON válido, sin markdown, con esta forma:
{
  "puntaje_global": 1,
  "nivel": "baja|media|alta",
  "resumen": "...",
  "dimensiones": {
    "completitud": {"puntaje": 1, "observacion": "..."},
    "accesibilidad": {"puntaje": 1, "observacion": "..."},
    "trazabilidad_documentacion": {"puntaje": 1, "observacion": "..."},
    "gobernanza": {"puntaje": 1, "observacion": "..."}
  },
  "brechas_criticas": ["..."],
  "recomendaciones": ["..."]
}

INICIATIVA:
{initiative_json}
"""

INNOVATION_FACTOR_PROMPT = """Eres un comité de innovación evaluando si una iniciativa aporta novedad real.

Considera la idea, el MVP, las bibliografías citadas y el contexto organizacional.

Devuelve solo JSON válido, sin markdown, con esta forma:
{
  "nivel": "bajo|medio|alto",
  "puntaje_sugerido": 1,
  "resumen": "...",
  "evidencias": ["..."],
  "brechas": ["..."],
  "recomendacion_ti": "..."
}

INICIATIVA:
{initiative_json}

BIBLIOGRAFÍA ANALIZADA:
{bibliografia_analisis}
"""

# Tool schemas for structured updates
class UpdateFormField(BaseModel):
    """Actualiza un campo del formulario de la iniciativa."""
    field_name: str = Field(..., description="El nombre técnico del campo (titulo, unidad, problema_oportunidad, resultado_esperado, mvp, datos_necesarios, datos_ubicacion, impacto_operacion, validacion_exito, beneficio_esperado, valor_estimado)")
    value: Union[str, Dict[str, Any]] = Field(..., description="Nuevo valor. Para 'beneficio_esperado' y 'valor_estimado' usa preferiblemente {'texto': '...', 'valor': numero}. 'valor_estimado' representa costo estimado.")

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

class BibliographyItem(BaseModel):
    """Una fuente bibliográfica del formulario."""
    url: str = Field(..., description="URL completa con http:// o https://")
    uso: str = Field("", description="Para qué se utilizará esta fuente en la iniciativa")

class UpdateBibliography(BaseModel):
    """Actualiza las URLs de bibliografía del formulario. Incluye al menos 3 fuentes válidas."""
    items: List[BibliographyItem] = Field(
        ...,
        description="Lista de fuentes [{url: 'https://...', uso: '...'}, ...]. Mínimo 3 URLs.",
    )

class _HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        self._in_title = tag.lower() == "title"

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data or "").strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text[:180]
        elif len(" ".join(self.parts)) < 1800:
            self.parts.append(text)


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


def _normalize_bibliography_items(items: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items or []:
        url = item.get("url") if isinstance(item, dict) else item
        url = str(url or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            normalized.append({"url": url, "dominio": parsed.netloc.lower()})
    return normalized


def _fetch_source_snapshot(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    snapshot: Dict[str, Any] = {
        "url": url,
        "dominio": parsed.netloc.lower(),
        "estado": "heuristica",
        "titulo": "",
        "extracto": "",
        "error": "",
    }
    try:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; IniciativaIA/1.0; +https://aldeaglobal.com)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            },
        )
        with urlopen(request, timeout=6) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read(120_000)
        text = raw.decode("utf-8", errors="ignore")
        if "html" in content_type.lower() or "<html" in text[:500].lower():
            parser = _HtmlTextExtractor()
            parser.feed(text)
            snapshot["titulo"] = parser.title
            snapshot["extracto"] = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()[:1800]
        else:
            snapshot["extracto"] = re.sub(r"\s+", " ", text).strip()[:1800]
        if snapshot["extracto"] or snapshot["titulo"]:
            snapshot["estado"] = "contenido_obtenido"
    except Exception as exc:
        snapshot["error"] = str(exc)[:220]
    return snapshot


def _fallback_bibliography_analysis(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    trusted_suffixes = (".edu", ".gov", ".gob", ".org")
    analyzed = []
    low_count = 0
    for source in sources:
        domain = str(source.get("dominio") or "").lower()
        has_content = source.get("estado") == "contenido_obtenido"
        reliability = "media"
        favor = []
        alerts = []
        if any(domain.endswith(suffix) or suffix in domain for suffix in trusted_suffixes):
            reliability = "alta"
            favor.append("Dominio institucional o de organización reconocible.")
        if has_content:
            favor.append("Se obtuvo contenido para revisar pertinencia.")
        else:
            alerts.append("No se pudo descargar contenido; evaluación basada en dominio y URL.")
        if not domain or "." not in domain:
            reliability = "baja"
            alerts.append("Dominio no verificable.")
        if reliability == "baja":
            low_count += 1
        analyzed.append(
            {
                "url": source.get("url"),
                "dominio": domain,
                "confiabilidad": reliability,
                "estado": source.get("estado") or "heuristica",
                "senales_favor": favor or ["URL con formato válido."],
                "senales_alerta": alerts,
                "recomendacion": "Complementar con una fuente institucional o primaria." if alerts else "Fuente utilizable como soporte, validar actualidad del contenido.",
            }
        )
    risk = "alto" if low_count else "medio" if any(item["confiabilidad"] == "media" for item in analyzed) else "bajo"
    return {
        "resumen": "Evaluación automática de fuentes basada en contenido disponible y señales del dominio.",
        "riesgo_general": risk,
        "fuentes": analyzed,
    }


def analyze_bibliography_sources(urls: Any, initiative_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    items = _normalize_bibliography_items(urls)
    snapshots = [_fetch_source_snapshot(item["url"]) for item in items]
    fallback = _fallback_bibliography_analysis(snapshots)
    if not snapshots:
        return fallback
    try:
        llm = get_llm()
        prompt = BIBLIOGRAPHY_RELIABILITY_PROMPT.replace(
            "{initiative_json}",
            json.dumps(initiative_payload or {}, ensure_ascii=False, indent=2),
        ).replace(
            "{sources_json}",
            json.dumps(snapshots, ensure_ascii=False, indent=2),
        )
        response = llm.invoke(
            prompt
        )
        raw = getattr(response, "content", response)
        parsed = json.loads(_strip_json_fence(str(raw)))
        if isinstance(parsed, dict) and isinstance(parsed.get("fuentes"), list):
            parsed.setdefault("resumen", fallback["resumen"])
            parsed.setdefault("riesgo_general", fallback["riesgo_general"])
            return parsed
    except Exception:
        pass
    return fallback


def bibliography_analysis_markdown(analysis: Any) -> str:
    if not isinstance(analysis, dict):
        return "No se recibió análisis de bibliografía."
    lines = [
        f"Resumen: {analysis.get('resumen') or 'Sin resumen disponible.'}",
        f"Riesgo general: {analysis.get('riesgo_general') or 'sin clasificar'}",
        "",
    ]
    for item in analysis.get("fuentes") or []:
        lines.append(
            "- "
            f"{item.get('url')} | confiabilidad: {item.get('confiabilidad', 'sin clasificar')} | "
            f"estado: {item.get('estado', 'sin estado')} | recomendación: {item.get('recomendacion', '')}"
        )
    return "\n".join(lines).strip()


def _fallback_data_quality_analysis(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = form_payload or {}
    completeness = 2
    if str(payload.get("datos_necesarios") or "").strip():
        completeness += 1
    if str(payload.get("datos_ubicacion") or "").strip():
        completeness += 1
    if str(payload.get("validacion_exito") or "").strip():
        completeness += 1
    completeness = min(5, completeness)
    accessibility = 3 if str(payload.get("datos_ubicacion") or "").strip() else 2
    traceability = 3 if isinstance(payload.get("bibliografia"), list) and len(payload.get("bibliografia") or []) >= 3 else 2
    governance = 2
    scores = [completeness, accessibility, traceability, governance]
    global_score = max(1, min(5, round(sum(scores) / len(scores))))
    level = "alta" if global_score >= 4 else "media" if global_score >= 3 else "baja"
    gaps = []
    if completeness < 3:
        gaps.append("Faltan detalles sobre datos necesarios o su validación.")
    if accessibility < 3:
        gaps.append("No queda claro dónde residen los datos ni cómo acceder a ellos.")
    if traceability < 3:
        gaps.append("La documentación o bibliografía no respalda suficientemente la propuesta.")
    return {
        "puntaje_global": global_score,
        "nivel": level,
        "resumen": "Evaluación heurística de calidad de data y documentación.",
        "dimensiones": {
            "completitud": {"puntaje": completeness, "observacion": "Basado en campos declarados del formulario."},
            "accesibilidad": {"puntaje": accessibility, "observacion": "Basado en ubicación de datos declarada."},
            "trazabilidad_documentacion": {"puntaje": traceability, "observacion": "Basado en bibliografía y validación."},
            "gobernanza": {"puntaje": governance, "observacion": "Sin evidencia explícita de owner o frecuencia de actualización."},
        },
        "brechas_criticas": gaps,
        "recomendaciones": ["Documentar fuente, formato, responsable y frecuencia de actualización de cada dataset."],
    }


def analyze_data_quality(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    fallback = _fallback_data_quality_analysis(form_payload)
    try:
        llm = get_llm()
        prompt = DATA_QUALITY_PROMPT.replace(
            "{initiative_json}",
            json.dumps(form_payload or {}, ensure_ascii=False, indent=2),
        )
        response = llm.invoke(prompt)
        parsed = _extract_json_object(str(getattr(response, "content", response)))
        if isinstance(parsed, dict) and parsed.get("puntaje_global") is not None:
            parsed.setdefault("resumen", fallback["resumen"])
            parsed.setdefault("nivel", fallback["nivel"])
            parsed.setdefault("dimensiones", fallback["dimensiones"])
            parsed.setdefault("brechas_criticas", fallback["brechas_criticas"])
            parsed.setdefault("recomendaciones", fallback["recomendaciones"])
            return parsed
    except Exception:
        pass
    return fallback


def data_quality_analysis_markdown(analysis: Any) -> str:
    if not isinstance(analysis, dict):
        return "No se recibió análisis de calidad de data."
    lines = [
        f"Resumen: {analysis.get('resumen') or 'Sin resumen.'}",
        f"Puntaje global: {analysis.get('puntaje_global', 'sin dato')} ({analysis.get('nivel') or 'sin nivel'})",
        "",
    ]
    dimensiones = analysis.get("dimensiones") or {}
    for key, item in dimensiones.items():
        if isinstance(item, dict):
            lines.append(f"- {key}: {item.get('puntaje', '?')} — {item.get('observacion', '')}")
    gaps = analysis.get("brechas_criticas") or []
    if gaps:
        lines.append("")
        lines.append("Brechas críticas:")
        for gap in gaps:
            lines.append(f"- {gap}")
    return "\n".join(lines).strip()


def _fallback_innovation_analysis(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    mvp = str(form_payload.get("mvp") or "").lower()
    score = 3
    if any(token in mvp for token in ("ia", "inteligencia artificial", "machine learning", "automat")):
        score = 4
    return {
        "nivel": "medio",
        "puntaje_sugerido": score,
        "resumen": "Evaluación heurística preliminar del grado de innovación.",
        "evidencias": ["Descripción del MVP disponible para revisión."],
        "brechas": ["Requiere validación de TI sobre novedad real frente a soluciones existentes."],
        "recomendacion_ti": "Confirmar si existe solución similar interna o en el mercado antes de priorizar.",
    }


def analyze_innovation_factor(
    form_payload: Dict[str, Any],
    bibliography_analysis: Any = None,
) -> Dict[str, Any]:
    fallback = _fallback_innovation_analysis(form_payload)
    try:
        llm = get_llm()
        prompt = INNOVATION_FACTOR_PROMPT.replace(
            "{initiative_json}",
            json.dumps(form_payload or {}, ensure_ascii=False, indent=2),
        ).replace(
            "{bibliografia_analisis}",
            json.dumps(bibliography_analysis or {}, ensure_ascii=False, indent=2),
        )
        response = llm.invoke(prompt)
        parsed = _extract_json_object(str(getattr(response, "content", response)))
        if isinstance(parsed, dict) and parsed.get("puntaje_sugerido") is not None:
            parsed.setdefault("resumen", fallback["resumen"])
            parsed.setdefault("nivel", fallback["nivel"])
            parsed.setdefault("evidencias", fallback["evidencias"])
            parsed.setdefault("brechas", fallback["brechas"])
            parsed.setdefault("recomendacion_ti", fallback["recomendacion_ti"])
            return parsed
    except Exception:
        pass
    return fallback


def innovation_analysis_markdown(analysis: Any) -> str:
    if not isinstance(analysis, dict):
        return "No se recibió evaluación de factor innovador."
    lines = [
        f"Resumen: {analysis.get('resumen') or 'Sin resumen.'}",
        f"Nivel: {analysis.get('nivel') or 'sin clasificar'} | Puntaje sugerido: {analysis.get('puntaje_sugerido', 'sin dato')}",
        f"Recomendación para TI: {analysis.get('recomendacion_ti') or 'Sin recomendación.'}",
    ]
    evidencias = analysis.get("evidencias") or []
    if evidencias:
        lines.append("")
        lines.append("Evidencias:")
        for item in evidencias:
            lines.append(f"- {item}")
    brechas = analysis.get("brechas") or []
    if brechas:
        lines.append("")
        lines.append("Brechas:")
        for item in brechas:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def generate_initiative_summary(form_payload: Dict[str, Any], bibliography_analysis: Any) -> str:
    try:
        llm = get_llm()
        prompt = INITIATIVE_SUMMARY_PROMPT.replace(
            "{initiative_json}",
            json.dumps(form_payload or {}, ensure_ascii=False, indent=2),
        ).replace(
            "{bibliografia_analisis}",
            json.dumps(bibliography_analysis or {}, ensure_ascii=False, indent=2),
        )
        response = llm.invoke(
            prompt
        )
        summary = str(getattr(response, "content", response) or "").strip()
        if summary:
            return summary
    except Exception:
        pass
    title = str(form_payload.get("titulo") or "La iniciativa").strip()
    problem = str(form_payload.get("problema_oportunidad") or "").strip()
    result = str(form_payload.get("resultado_esperado") or "").strip()
    benefit = money_field_text(form_payload.get("beneficio_esperado"))
    return (
        f"{title} busca atender: {problem or 'una oportunidad de mejora operativa'}. "
        f"El resultado esperado es {result or 'mejorar eficiencia y control del proceso'}. "
        f"El beneficio principal es {benefit or 'generar valor medible para la operación'}."
    )


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
        )
    ).strip()
    benefit_text = money_field_text(data.get("beneficio_esperado"))
    if benefit_text:
        strategic_query = f"{strategic_query}\n{benefit_text}".strip()
    bibliography_analysis = data.get("bibliografia_analisis")
    bibliography_context = bibliography_analysis_markdown(bibliography_analysis)
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
            "beneficio_esperado", "valor_estimado", "dias_implementacion_estimados",
            "kpis_str", "bibliografia_context", "strategic_context",
            "calidad_datos_context", "factor_innovador_context",
        ],
        template=ANALYSIS_PROMPT
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    dias = data.get("dias_implementacion_estimados")
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
        "beneficio_esperado": beneficio_field_text(data.get("beneficio_esperado")),
        "valor_estimado": money_field_prompt(data.get("valor_estimado"), "costo"),
        "dias_implementacion_estimados": dias if dias is not None else "Sin estimación",
        "kpis_str": kpis_str,
        "bibliografia_context": bibliography_context,
        "strategic_context": strategic_context,
        "calidad_datos_context": data_quality_analysis_markdown(data.get("calidad_datos_analisis")),
        "factor_innovador_context": innovation_analysis_markdown(data.get("factor_innovador_analisis")),
    })
    return _normalize_analysis_markdown(result["text"], data.get("titulo"))


def _extract_json_object(text: str) -> Dict[str, Any]:
    body = (text or "").strip()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", body, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _score_fields_from_response(response: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, str], str]:
    criteria = response.get("criterios") if isinstance(response.get("criterios"), dict) else response
    scores: Dict[str, Any] = {}
    explanations: Dict[str, Any] = {}

    for criterion in BUSINESS_CRITERION_KEYS:
        value = criteria.get(criterion) if isinstance(criteria, dict) else None
        if isinstance(value, dict):
            scores[criterion] = value.get("puntaje", value.get("score"))
            explanations[criterion] = value.get("explicacion", value.get("comentario", ""))
        else:
            scores[criterion] = value

    resumen = response.get("resumen") if isinstance(response.get("resumen"), str) else ""
    return (
        normalize_business_scores(scores),
        normalize_business_explanations(explanations),
        " ".join(resumen.strip().split()),
    )


def suggest_business_score(data: dict, analysis: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt_template = PromptTemplate(
        input_variables=["initiative_json", "analysis"],
        template=SCORE_SUGGESTION_PROMPT,
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke(
        {
            "initiative_json": json.dumps(data, ensure_ascii=False, indent=2),
            "analysis": analysis or "",
        }
    )
    response = _extract_json_object(result["text"])
    normalized, explanations, resumen = _score_fields_from_response(response)
    criteria_response = response.get("criterios") if isinstance(response.get("criterios"), dict) else response
    if not any(key in criteria_response for key in BUSINESS_CRITERION_KEYS):
        normalized = default_business_scores()
    days = effective_implementation_days(data)
    adjusted_score, adjusted_explanation = apply_time_to_value_from_days(
        normalized.get("time_to_value", 3),
        days,
        explanations.get("time_to_value", ""),
    )
    normalized["time_to_value"] = adjusted_score
    explanations["time_to_value"] = adjusted_explanation
    calidad = data.get("calidad_datos_analisis") if isinstance(data.get("calidad_datos_analisis"), dict) else {}
    global_data_score = calidad.get("puntaje_global")
    try:
        global_data_score = int(global_data_score)
    except (TypeError, ValueError):
        global_data_score = None
    if global_data_score is not None and global_data_score <= 2:
        normalized["datos_ia"] = min(normalized.get("datos_ia", 3), 2)
        explanations["datos_ia"] = (
            f"{explanations.get('datos_ia', '')} Limitado por calidad de data ({global_data_score}/5)."
        ).strip()
    payload = build_potenciadores_payload(
        normalized,
        estado="sugerido",
        fuente="ia",
        comentario="Sugerencia generada automáticamente por IA.",
        explanations=explanations,
        resumen=resumen or "Score sugerido según impacto y agilidad.",
    )
    return apply_implementation_days_to_potenciadores(payload, days)


def suggest_roi_probability(data: dict, analysis: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt_template = PromptTemplate(
        input_variables=["initiative_json", "analysis"],
        template=ROI_PROBABILITY_PROMPT,
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke(
        {
            "initiative_json": json.dumps(data, ensure_ascii=False, indent=2),
            "analysis": analysis or "",
        }
    )
    response = _extract_json_object(result["text"])
    fallback = estimate_success_probability(data)
    try:
        probability = float(response.get("p_exito", fallback["p_exito"]))
    except (TypeError, ValueError):
        probability = fallback["p_exito"]
    if probability > 1:
        probability = probability / 100
    probability = round(max(0.0, min(1.0, probability)), 2)
    explanation = " ".join(str(response.get("explicacion") or fallback["explicacion"]).split())
    return {"p_exito": probability, "explicacion": explanation}


def append_score_to_analysis(analysis: str, potenciadores: Dict[str, Any]) -> str:
    return analysis

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
    "valor_estimado",
    "beneficio_esperado",
]

_FIELD_PROPOSAL_MARKERS = {
    "problema_oportunidad": ("problema u oportunidad", "problema o oportunidad"),
    "resultado_esperado": ("resultado esperado",),
    "titulo": ("titulo", "título", "nombre de la iniciativa"),
    "unidad": ("unidad de negocio",),
    "mvp": ("mvp", "primera version", "primera versión", "producto minimo viable", "producto mínimo viable"),
    "datos_necesarios": ("datos necesarios",),
    "datos_ubicacion": (
        "ubicacion de datos",
        "ubicación de datos",
        "donde estan los datos",
        "donde están los datos",
    ),
    "impacto_operacion": (
        "impacto operativo",
        "impacto en la operacion",
        "impacto en la operación",
    ),
    "validacion_exito": (
        "validacion del exito",
        "validación del éxito",
        "validacion de exito",
        "validación de éxito",
        "como validar",
        "cómo validar",
    ),
    "beneficio_esperado": ("beneficio esperado", "beneficio cualitativo"),
    "valor_estimado": ("costo estimado", "valor estimado"),
    "bibliografia": (
        "bibliografia",
        "bibliografía",
        "fuentes",
        "urls de referencia",
        "articulos",
        "artículos",
        "referencias",
    ),
}

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
    if key == "bibliografia":
        return _bibliography_needs_fill({"bibliografia": value})
    if key == "beneficio_esperado":
        return _beneficio_field_target(value) is not None
    if key == "valor_estimado":
        return _money_field_target(key, value) is not None
    if isinstance(value, (int, float)):
        return False
    text = str(value).strip()
    if not text:
        return True
    if _is_conversational_feedback_message(text):
        return True
    return False


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

    positive_phrases = (
        "me parece bien",
        "me parece excelente",
        "si me parece excelente",
        "este si me parece excelente",
        "me gusta",
        "lo dejamos asi",
        "lo dejamos así",
        "si asi dejemoslo",
        "si así dejemoslo",
        "asi dejemoslo",
        "así dejemoslo",
        "dejemoslo asi",
        "dejémoslo así",
    )
    if any(phrase in normalized_simple for phrase in positive_phrases):
        return True

    if normalized_simple.startswith("no ") or normalized_simple.startswith("no,"):
        return False

    if normalized in _GENERIC_USER_CONFIRMATION_PHRASES or normalized_simple in _GENERIC_USER_CONFIRMATION_PHRASES:
        return True

    fragments = (
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
    if "bien" in tokens and "si" in tokens:
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
    candidates: List[str] = []
    quote_pairs = [("“", "”"), ('"', '"'), ("'", "'")]
    for left, right in quote_pairs:
        start = 0
        while True:
            start = raw.find(left, start)
            if start == -1:
                break
            end = raw.find(right, start + 1)
            if end == -1:
                break
            candidate = raw[start + 1 : end].strip()
            if candidate:
                candidates.append(candidate)
            start = end + 1
    if not candidates:
        return ""
    return max(candidates, key=len)


def _content_mentions_scalar_field(content: str, field_name: str) -> bool:
    normalized = _normalize_text_for_match(content)
    if not normalized or not field_name:
        return False
    markers = _FIELD_PROPOSAL_MARKERS.get(field_name, ())
    return any(_normalize_text_for_match(marker) in normalized for marker in markers)


def _infer_confirmed_target_from_assistant(
    history: List[dict],
) -> Optional[Union[Tuple[str, str], Tuple[str, int, str]]]:
    content = _get_last_assistant_message(history)
    if not content:
        return None
    normalized = _normalize_text_for_match(content)
    priority = (
        "mvp",
        "titulo",
        "resultado_esperado",
        "unidad",
        "datos_necesarios",
        "datos_ubicacion",
        "impacto_operacion",
        "validacion_exito",
        "beneficio_esperado",
        "valor_estimado",
        "bibliografia",
    )
    for field_name in priority:
        if field_name == "bibliografia":
            if _extract_urls_from_text(content):
                return ("scalar", "bibliografia")
            continue
        if _content_mentions_scalar_field(content, field_name):
            return ("scalar", field_name)
    if any(token in normalized for token in ("kpi", "indicador", "linea base", "línea base", "meta")):
        return ("kpi", 0, "indicador")
    return None


def _proposal_from_assistant_text(
    content: str,
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> Optional[str]:
    if not target or not content:
        return None
    if target[0] == "kpi":
        _kind, _row_idx, sub = target
        low = _normalize_text_for_match(content)
        markers = {
            "indicador": ("indicador", "kpi", "metrica", "métrica"),
            "base": ("linea base", "línea base", "base", "situacion actual", "situación actual"),
            "meta": ("meta", "objetivo", "valor deseado"),
        }
        sub_markers = markers.get(sub, ())
        if sub_markers and not any(marker in low for marker in sub_markers):
            return None
        extracted = _extract_text_between_quotes(content)
        return extracted or None
    if target[0] == "money":
        field_name, sub = target[1], target[2]
        parsed = _extract_embedded_dict_from_content(content)
        if parsed:
            if field_name == "beneficio_esperado":
                if sub == "texto" and parsed.get("texto"):
                    return str(parsed["texto"])
                if sub == "comunicacion" and parsed.get("comunicacion"):
                    return str(parsed["comunicacion"])
                if sub == "valor" and parsed.get("valor") is not None:
                    return str(parsed["valor"])
            elif field_name == "valor_estimado":
                if sub == "texto" and parsed.get("texto"):
                    return str(parsed["texto"])
                if sub == "valor" and parsed.get("valor") is not None:
                    return str(parsed["valor"])
        extracted = _extract_text_between_quotes(content)
        if extracted:
            return extracted
        if sub == "valor":
            from app.roi import _parse_number

            for match in re.finditer(r"\d[\d.,]*", content):
                parsed_number = _parse_number(match.group())
                if parsed_number is not None:
                    return str(parsed_number)
        return None
    field_name = target[1]
    if not _content_mentions_scalar_field(content, field_name):
        return None
    extracted = _extract_text_between_quotes(content)
    if extracted:
        return extracted
    if field_name in {"beneficio_esperado", "valor_estimado"}:
        parsed = _extract_embedded_dict_from_content(content)
        if parsed:
            if field_name == "beneficio_esperado" and parsed.get("texto"):
                return str(parsed["texto"])
            if field_name == "valor_estimado" and parsed.get("texto"):
                return str(parsed["texto"])
        return content.strip() or None
    return None


def _proposal_from_last_assistant(
    history: List[dict],
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> Optional[str]:
    if not target:
        return None
    content = _get_last_assistant_message(history).strip()
    if not content:
        return None
    inferred = _proposal_from_assistant_text(content, target)
    if inferred:
        return inferred
    if target[0] not in {"scalar", "money"}:
        return None
    if target[0] == "scalar" and _infer_confirmed_target_from_assistant(history) != target:
        return None
    extracted = _extract_text_between_quotes(content)
    return extracted or None


def _extract_proposal_for_confirmation(
    history: List[dict],
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> Optional[str]:
    return _proposal_from_last_assistant(history, target)


def _upsert_scalar_form_update(
    form_updates: List[Dict[str, Any]],
    field_name: str,
    value: Any,
) -> None:
    kept = [
        update
        for update in form_updates
        if not (
            update.get("function") == "UpdateFormField"
            and (update.get("args") or {}).get("field_name") == field_name
        )
    ]
    form_updates[:] = kept
    form_updates.append(
        {
            "function": "UpdateFormField",
            "args": {
                "field_name": field_name,
                "value": value,
            },
        }
    )


def _get_last_assistant_message(history: List[dict]) -> str:
    for msg in reversed(history or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("assistant", "ai", "agent"):
            return str(msg.get("content") or "")
    return ""


def _extract_embedded_dict_from_content(content: str) -> Optional[dict]:
    text = (content or "").strip()
    if not text or "{" not in text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        return None
    chunk = text[start : end + 1]
    for parser in (ast.literal_eval, json.loads):
        try:
            candidate = chunk if parser is ast.literal_eval else chunk.replace("'", '"')
            obj = parser(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _humanize_chat_response(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return text
    parsed = _extract_embedded_dict_from_content(text)
    if not parsed:
        return text

    labels = {
        "texto": "Descripción",
        "comunicacion": "Comunicación que facilita",
        "valor": "Valor ($)",
    }
    lines: List[str] = []
    for key in ("texto", "comunicacion", "valor"):
        if key not in parsed or parsed[key] in (None, ""):
            continue
        val = parsed[key]
        if key == "valor":
            try:
                val = f"${float(val):,.0f}".replace(",", ".")
            except (TypeError, ValueError):
                val = str(val)
        lines.append(f"- **{labels[key]}:** {val}")
    if not lines:
        return text

    start = text.find("{")
    end = text.rfind("}")
    before = text[:start].strip()
    after = text[end + 1 :].strip()
    body = "\n".join(lines)
    parts = [part for part in (before, body, after) if part]
    return "\n\n".join(parts)


def _resolve_guided_confirmation_target(
    history: List[dict],
    merged_before: dict,
    target_before: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
    message: str,
) -> Optional[Union[Tuple[str, str], Tuple[str, int, str]]]:
    if not _is_affirmative_message(message):
        return target_before
    # Mantener el paso activo del flujo guiado; el mensaje del asistente suele mencionar otros campos.
    if target_before is not None:
        return target_before
    inferred = _infer_confirmed_target_from_assistant(history)
    if inferred:
        return inferred
    return target_before


def _append_guided_numeric_fallback(
    form_updates: List[Dict[str, Any]],
    *,
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
    message: str,
) -> None:
    if not target or target[0] != "money" or target[2] != "valor":
        return
    if _is_affirmative_message(message):
        return
    from app.roi import _parse_number

    parsed = _parse_number((message or "").strip())
    if parsed is None:
        return
    field_name = target[1]
    _upsert_scalar_form_update(form_updates, field_name, {"valor": parsed})


def _has_update_for_target(
    form_updates: List[Dict[str, Any]],
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> bool:
    if not target:
        return False
    if target[0] == "scalar":
        field_name = target[1]
        for update in form_updates or []:
            if update.get("function") != "UpdateFormField":
                continue
            args = update.get("args") or {}
            if args.get("field_name") != field_name:
                continue
            value = args.get("value")
            if value is None or value == "":
                continue
            return True
        return False
    if target[0] == "kpi":
        return any((update.get("function") == "UpdateKpis") for update in (form_updates or []))
    if target[0] == "money":
        field_name = target[1]
        for update in form_updates or []:
            if update.get("function") != "UpdateFormField":
                continue
            args = update.get("args") or {}
            if args.get("field_name") != field_name:
                continue
            value = args.get("value")
            if isinstance(value, dict) and value.get("valor") is not None:
                return True
        return False
    return False


def _append_guided_confirmation_fallback(
    form_updates: List[Dict[str, Any]],
    *,
    history: List[dict],
    merged_before: dict,
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
    message: str,
) -> None:
    if _append_guided_unit_correction_fallback(
        form_updates,
        history=history,
        target_before=target,
        message=message,
    ):
        return
    if not target or not _is_affirmative_message(message):
        return

    if target[0] == "kpi":
        _kind, row_idx, sub = target
        inferred_value = _extract_proposal_for_confirmation(history, target)
        if not inferred_value:
            return
        kpis = copy.deepcopy(merged_before.get("kpis") or [{"indicador": "", "base": "", "meta": ""}])
        while len(kpis) <= row_idx:
            kpis.append({"indicador": "", "base": "", "meta": ""})
        if not isinstance(kpis[row_idx], dict):
            kpis[row_idx] = {"indicador": "", "base": "", "meta": ""}
        kpis[row_idx][sub] = inferred_value
        form_updates[:] = [update for update in form_updates if update.get("function") != "UpdateKpis"]
        form_updates.append(
            {
                "function": "UpdateKpis",
                "args": {"items": _merge_kpi_items(merged_before.get("kpis") or [], kpis)},
            }
        )
        return

    if target[0] == "money":
        field_name, sub = target[1], target[2]
        assistant_content = _get_last_assistant_message(history)
        parsed_dict = _extract_embedded_dict_from_content(assistant_content)
        if parsed_dict and field_name == "beneficio_esperado":
            has_data = any(
                parsed_dict.get(key) not in (None, "")
                for key in ("texto", "comunicacion", "valor")
            )
            if has_data:
                _upsert_scalar_form_update(form_updates, field_name, parsed_dict)
                return
        inferred_value = _extract_proposal_for_confirmation(history, target)
        if not inferred_value:
            return
        if field_name == "beneficio_esperado":
            if sub == "valor":
                from app.roi import _parse_number
                parsed = _parse_number(inferred_value)
                value = {"valor": parsed if parsed is not None else inferred_value}
            elif sub == "comunicacion":
                value = {"comunicacion": inferred_value}
            else:
                value = {"texto": inferred_value}
        else:
            if sub == "valor":
                from app.roi import _parse_number
                parsed = _parse_number(inferred_value)
                value = {"valor": parsed if parsed is not None else inferred_value}
            else:
                value = {"texto": inferred_value}
        _upsert_scalar_form_update(form_updates, field_name, value)
        return

    if target[0] != "scalar" or _message_mentions_kpi_instruction(message):
        return

    field_name = target[1]
    if field_name == "bibliografia":
        urls = _collect_bibliography_urls_from_history(history)
        if len(urls) >= 3:
            _upsert_bibliography_update(form_updates, [{"url": url} for url in urls])
        return

    inferred_value = _extract_proposal_for_confirmation(history, target)
    if not inferred_value:
        return
    _upsert_scalar_form_update(form_updates, field_name, inferred_value)


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
    if any(k in blob for k in ("café", "cafe", "coffee", "taza", "barismo", "barista", "granos")):
        return "Aldea Coffee"
    if any(k in blob for k in ("zon", "aldeazon", "amazon", "marketplace", "ventas en línea", "ventas en linea", "venta", "comercio electrónico", "comercio electronico")):
        return "AldeaZON"
    if any(k in blob for k in ("fundación", "fundacion", "social", "comunidad", "donación", "donaciones", "ayuda", "beneficio", "ambiental")):
        return "Fundación Aldea"
    if any(k in blob for k in ("contabilidad", "contable", "finanzas", "costo", "costos", "factura", "facturación", "facturacion", "impuesto", "auditoría", "auditoria", "balance", "presupuesto", "gasto", "gastos")):
        return "Contabilidad"
    if any(k in blob for k in ("talento", "rrhh", "humano", "personal", "reclutamiento", "contratación", "contratacion", "planilla", "nómina", "nomina", "capacitación", "capacitacion")):
        return "Talento Humano"
    if any(k in blob for k in ("certificación", "certificacion", "calidad", "norma", "iso", "estándar", "estandar")):
        return "Certificación"
    if any(k in blob for k in ("ti", "sistemas", "tecnología", "tecnologia", "desarrollo", "software", "computación", "redes", "it", "programación", "programacion", "soporte técnico", "soporte tecnico")):
        return "TI"
    if any(k in blob for k in ("producto", "productos", "inventario", "stock", "bodega", "almacén", "almacen")):
        return "Productos"
    return "Aldea Global"


_VALID_BUSINESS_UNITS = (
    "Aldea Global",
    "Aldea Coffee",
    "AldeaZON",
    "Fundación Aldea",
    "Contabilidad",
    "Talento Humano",
    "Certificación",
    "TI",
    "Productos",
)

_BUSINESS_UNIT_ALIASES = {
    "aldea global": "Aldea Global",
    "aldea coffee": "Aldea Coffee",
    "coffee": "Aldea Coffee",
    "cafe": "Aldea Coffee",
    "café": "Aldea Coffee",
    "aldeazon": "AldeaZON",
    "zon": "AldeaZON",
    "fundacion aldea": "Fundación Aldea",
    "fundación aldea": "Fundación Aldea",
    "fundacion": "Fundación Aldea",
    "contabilidad": "Contabilidad",
    "conta": "Contabilidad",
    "finanzas": "Contabilidad",
    "talento humano": "Talento Humano",
    "rrhh": "Talento Humano",
    "humano": "Talento Humano",
    "certificacion": "Certificación",
    "certificación": "Certificación",
    "ti": "TI",
    "it": "TI",
    "tecnologia": "TI",
    "tecnología": "TI",
    "sistemas": "TI",
    "productos": "Productos",
}


def _normalize_business_unit(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = _normalize_text_for_match(text)
    for unit in _VALID_BUSINESS_UNITS:
        if _normalize_text_for_match(unit) == normalized:
            return unit
    return _BUSINESS_UNIT_ALIASES.get(normalized)


def _match_business_unit_in_text(text: str) -> Optional[str]:
    normalized = _normalize_text_for_match(text)
    if not normalized:
        return None
    for unit in sorted(_VALID_BUSINESS_UNITS, key=len, reverse=True):
        unit_norm = _normalize_text_for_match(unit)
        if unit_norm in normalized:
            return unit
    for alias, canonical in sorted(_BUSINESS_UNIT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if len(alias) <= 3:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
        elif alias in normalized:
            return canonical
    return None


def _extract_business_unit_from_message(message: str) -> Optional[str]:
    raw = (message or "").strip()
    if not raw:
        return None
    direct = _match_business_unit_in_text(raw)
    if direct:
        return direct
    for segment in re.split(r"[,;]", raw):
        match = _match_business_unit_in_text(segment.strip())
        if match:
            return match
    normalized = _normalize_text_for_match(raw)
    tail_patterns = (
        r"(?:ponle|pon|es|sea|seria|sería|debe ser|realmente es|mejor|unidad|area|área)\s+(.+)$",
    )
    for pattern in tail_patterns:
        found = re.search(pattern, normalized)
        if not found:
            continue
        match = _match_business_unit_in_text(found.group(1))
        if match:
            return match
    return None


def _is_unit_correction_context(
    history: List[dict],
    target_before: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
) -> bool:
    if target_before == ("scalar", "unidad"):
        return True
    if _infer_confirmed_target_from_assistant(history) == ("scalar", "unidad"):
        return True
    return "unidad de negocio" in _get_last_assistant_message(history).lower()


def _append_guided_unit_correction_fallback(
    form_updates: List[Dict[str, Any]],
    *,
    history: List[dict],
    target_before: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
    message: str,
) -> bool:
    unit = _extract_business_unit_from_message(message)
    if not unit or not _is_unit_correction_context(history, target_before):
        return False
    if _has_update_for_target(form_updates, ("scalar", "unidad")):
        for update in form_updates:
            if update.get("function") != "UpdateFormField":
                continue
            args = update.get("args") or {}
            if args.get("field_name") == "unidad":
                args["value"] = unit
        return True
    form_updates.append(
        {
            "function": "UpdateFormField",
            "args": {
                "field_name": "unidad",
                "value": unit,
            },
        }
    )
    return True


def _build_unit_correction_content(
    current_form: Optional[dict],
    form_updates: List[Dict[str, Any]],
) -> Optional[str]:
    merged_after = _apply_form_updates(current_form, form_updates)
    unit = _normalize_business_unit(merged_after.get("unidad"))
    if not unit:
        return None
    nxt = _next_guided_target(merged_after)
    next_line = _guided_specific_proposal(nxt, merged_after) or _guided_question(nxt)
    return f'Perfecto, dejamos la unidad de negocio en “{unit}”. {next_line}'


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
    current = normalize_beneficio_field(merged.get("beneficio_esperado")).get("texto")
    if current:
        return current
    return "Mejor calidad de servicio, mayor confianza en la información y decisiones más rápidas basadas en evidencia."


def _suggest_value_from_form(merged: dict) -> str:
    current = money_field_text(merged.get("valor_estimado"))
    if current:
        return current
    return "USD 12,000 estimados, considerando desarrollo, integraciones, acompañamiento y soporte inicial."


def _money_field_parts(value: Any) -> Dict[str, Any]:
    normalized = normalize_money_field(value)
    return {
        "texto": str(normalized.get("texto") or "").strip(),
        "valor": normalized.get("valor"),
    }


def _merge_money_field(existing: Any, incoming: Any) -> Dict[str, Any]:
    current = _money_field_parts(existing)
    new_value = _money_field_parts(incoming)
    return {
        "texto": new_value["texto"] or current["texto"],
        "valor": new_value["valor"] if new_value["valor"] is not None else current["valor"],
    }


def _money_field_target(field_name: str, value: Any) -> Optional[Tuple[str, str, str]]:
    parts = _money_field_parts(value)
    if not parts["texto"]:
        return ("money", field_name, "texto")
    if parts["valor"] is None:
        return ("money", field_name, "valor")
    return None


def _beneficio_field_parts(value: Any) -> Dict[str, Any]:
    normalized = normalize_beneficio_field(value)
    return {
        "texto": str(normalized.get("texto") or "").strip(),
        "comunicacion": str(normalized.get("comunicacion") or "").strip(),
        "valor": normalized.get("valor"),
    }


def _merge_beneficio_field(existing: Any, incoming: Any) -> Dict[str, Any]:
    current = _beneficio_field_parts(existing)
    new_value = _beneficio_field_parts(incoming)
    return {
        "texto": new_value["texto"] or current["texto"],
        "comunicacion": new_value["comunicacion"] or current["comunicacion"],
        "valor": new_value["valor"] if new_value["valor"] is not None else current["valor"],
    }


def _beneficio_field_target(value: Any) -> Optional[Tuple[str, str, str]]:
    parts = _beneficio_field_parts(value)
    if not parts["texto"]:
        return ("money", "beneficio_esperado", "texto")
    if not parts["comunicacion"]:
        return ("money", "beneficio_esperado", "comunicacion")
    if parts["valor"] is None:
        return ("money", "beneficio_esperado", "valor")
    return None


def _guided_specific_proposal(
    target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]], merged: dict
) -> Optional[str]:
    if not target:
        return None
    if target[0] == "scalar":
        key = target[1]
        if not _is_empty_field(key, merged.get(key)):
            return None
    if target[0] != "scalar":
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
        return f"Te propongo este beneficio esperado: “{benefit}”. Indícame también el valor monetario estimado para calcular el ROI. ¿Lo dejamos así?"
    if key == "valor_estimado":
        value = _suggest_value_from_form(merged)
        return (
            "Te propongo este costo estimado con contexto de negocio: "
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


def _is_conversational_feedback_message(message: str) -> bool:
    normalized = _normalize_text_for_match(message)
    if not normalized:
        return False

    positive_phrases = (
        "me parece bien",
        "me parece excelente",
        "si me parece excelente",
        "este si me parece excelente",
        "me gusta",
        "si asi dejemoslo",
        "si así dejemoslo",
    )
    if any(phrase in normalized for phrase in positive_phrases):
        return False

    if normalized.startswith("no ") or normalized.startswith("no,"):
        return True

    markers = (
        "no me parece",
        "me parece que",
        "ponle",
        "pongas",
        "le pongas",
        "algo asi",
        "o algo asi",
        "en vez de",
        "mejor que",
        "cambialo",
        "cambiala",
        "ajustalo",
        "ajustala",
        "no quiero",
        "prefiero que",
        "no es para nada",
        "no es mi tema",
    )
    return any(marker in normalized for marker in markers)


def _value_echoes_user_message(value: Any, message: str) -> bool:
    if isinstance(value, dict):
        parts = [
            str(value.get("texto") or ""),
            str(value.get("valor") or ""),
        ]
        text = " ".join(part for part in parts if part).strip()
    else:
        text = str(value or "").strip()
    normalized_value = _normalize_text_for_match(text)
    normalized_message = _normalize_text_for_match(message)
    if not normalized_value or not normalized_message:
        return False
    if normalized_value == normalized_message:
        return True
    if len(normalized_message) >= 24 and normalized_message in normalized_value:
        return True
    if len(normalized_value) >= 24 and normalized_value in normalized_message:
        return True
    return False


def _strip_invalid_guided_form_updates(
    form_updates: List[Dict[str, Any]],
    message: str,
    *,
    guided_mode: bool,
) -> List[Dict[str, Any]]:
    if not guided_mode:
        return form_updates

    cleaned: List[Dict[str, Any]] = []
    for update in form_updates or []:
        name = (update.get("function") or update.get("name") or "").strip()
        if name != "UpdateFormField":
            cleaned.append(update)
            continue
        args = update.get("args") or {}
        value = args.get("value")
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            has_text = bool(str(value.get("texto") or "").strip())
            has_number = value.get("valor") not in (None, "")
            if not has_text and not has_number:
                continue
        text_value = value.get("texto") if isinstance(value, dict) else str(value or "")
        if _is_conversational_feedback_message(text_value):
            continue
        if _value_echoes_user_message(value, message):
            continue
        if _is_conversational_feedback_message(message) and _value_echoes_user_message(value, message):
            continue
        cleaned.append(update)
    return cleaned


def _extract_user_value_for_target(field_name: str, message: str) -> Optional[str]:
    raw = (message or "").strip()
    if not raw:
        return None
    if (
        _is_meta_instruction_message(raw)
        or _message_mentions_kpi_instruction(raw)
        or _is_conversational_feedback_message(raw)
    ):
        return None
    normalized_raw = _normalize_text_for_match(raw)
    if (
        _is_affirmative_message(raw)
        or normalized_raw in _GENERIC_USER_CONFIRMATION_PHRASES
        or len(normalized_raw) <= 18 and normalized_raw in _GENERIC_USER_CONFIRMATION_PHRASES
    ):
        return None
    if len(raw) > 80:
        return None
    if field_name in {"beneficio_esperado", "valor_estimado"}:
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
        "beneficio_esperado": ("te propongo el beneficio esperado", "te propongo el beneficio cualitativo"),
        "valor_estimado": ("si tienes una cifra, dime el costo estimado", "si tienes una cifra, dime el valor estimado"),
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


_MAX_KPI_ROWS = 10
_URL_PATTERN = re.compile(r"https?://[^\s\)\]>\'\"<,]+", re.IGNORECASE)


def _extract_urls_from_text(text: str) -> List[str]:
    urls: List[str] = []
    seen: set = set()
    for match in _URL_PATTERN.findall(text or ""):
        url = match.rstrip(".,;:)")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _valid_bibliography_urls(items: Any) -> List[str]:
    valid: List[str] = []
    for item in items or []:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
        else:
            continue
        if not url:
            continue
        try:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                valid.append(url)
        except Exception:
            continue
    return valid


def _bibliography_needs_fill(merged: dict) -> bool:
    return len(_valid_bibliography_urls(merged.get("bibliografia"))) < 3


def _collect_bibliography_urls_from_history(history: List[dict]) -> List[str]:
    urls: List[str] = []
    seen: set = set()
    for msg in reversed(history or []):
        if msg.get("role") != "agent":
            continue
        for url in _extract_urls_from_text(str(msg.get("content") or "")):
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _normalize_bibliography_items(items: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for item in items or []:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(item.get("url") or "").strip()
        else:
            continue
        if url:
            uso = ""
            if isinstance(item, dict):
                uso = str(item.get("uso") or item.get("proposito") or "").strip()
            rows.append({"url": url, "uso": uso})
    while len(rows) < 3:
        rows.append({"url": "", "uso": ""})
    return rows[:10]


def _upsert_bibliography_update(
    form_updates: List[Dict[str, Any]],
    items: List[dict],
) -> None:
    kept = [
        update
        for update in form_updates
        if (update.get("function") or update.get("name")) != "UpdateBibliography"
    ]
    form_updates[:] = kept
    form_updates.append(
        {
            "function": "UpdateBibliography",
            "args": {"items": _normalize_bibliography_items(items)},
        }
    )


def _is_bibliography_add_request(message: str) -> bool:
    normalized = _normalize_text_for_match(message)
    if not normalized:
        return False
    markers = (
        "agregala",
        "agregalas",
        "agreguelas",
        "añadelas",
        "anadelas",
        "ponlas",
        "incluyelas",
        "incorporalas",
        "guardalas",
        "usalas",
        "colocalas",
        "agrega esas",
        "agrega las",
        "pon esas",
    )
    return any(marker in normalized for marker in markers)


def _append_bibliography_confirmation_fallback(
    form_updates: List[Dict[str, Any]],
    *,
    history: List[dict],
    message: str,
) -> bool:
    should_apply = _is_bibliography_add_request(message) or (
        _is_affirmative_message(message)
        and _content_mentions_scalar_field(_get_last_assistant_message(history), "bibliografia")
    )
    if not should_apply:
        return False
    urls = _collect_bibliography_urls_from_history(history)
    if len(urls) < 3:
        return False
    _upsert_bibliography_update(form_updates, [{"url": url} for url in urls])
    return True

_KPI_INSTRUCTION_MARKERS = (
    "kpi",
    "kpis",
    "indicador",
    "linea base",
    "línea base",
    "agrega",
    "agregar",
    "anade",
    "añade",
    "añadir",
    "pon ",
    "ponle",
    "modifica",
    "actualiza",
    "cambia",
    "editar",
    "corrige",
    "otro kpi",
    "mas kpi",
    "más kpi",
    "nuevo kpi",
)

_INSTRUCTION_VERBS = (
    "agrega",
    "agregar",
    "anade",
    "añade",
    "añadir",
    "pon",
    "ponle",
    "modifica",
    "actualiza",
    "cambia",
    "editar",
    "corrige",
    "completa",
    "llena",
)


def _normalize_kpi_row(item: Any) -> dict:
    if not isinstance(item, dict):
        return {"indicador": "", "base": "", "meta": ""}
    indicador = str(item.get("indicador") or item.get("metrica") or "").strip()
    base = str(
        item.get("base") or item.get("linea_base") or item.get("lineaBase") or ""
    ).strip()
    meta = str(
        item.get("meta") or item.get("meta_deseada") or item.get("metaDeseada") or ""
    ).strip()
    return {"indicador": indicador, "base": base, "meta": meta}


def _is_kpi_row_empty(row: dict) -> bool:
    return not (row.get("indicador") or row.get("base") or row.get("meta"))


def _sanitize_kpi_list(rows: List[dict]) -> List[dict]:
    normalized = [_normalize_kpi_row(row) for row in (rows or [])]
    non_empty = [row for row in normalized if not _is_kpi_row_empty(row)]
    if non_empty:
        return non_empty[:_MAX_KPI_ROWS]
    return [{"indicador": "", "base": "", "meta": ""}]


def _merge_kpi_items(existing: List[dict], incoming: List[dict]) -> List[dict]:
    existing_norm = [_normalize_kpi_row(row) for row in (existing or [])]
    incoming_norm = [_normalize_kpi_row(row) for row in (incoming or [])]
    if not incoming_norm:
        return _sanitize_kpi_list(existing_norm)

    merged: List[dict] = []
    max_len = max(len(existing_norm), len(incoming_norm))
    for index in range(max_len):
        base_row = (
            existing_norm[index]
            if index < len(existing_norm)
            else {"indicador": "", "base": "", "meta": ""}
        )
        incoming_row = incoming_norm[index] if index < len(incoming_norm) else None
        if incoming_row is None:
            merged.append(dict(base_row))
            continue
        row = dict(base_row)
        for key in ("indicador", "base", "meta"):
            if incoming_row.get(key):
                row[key] = incoming_row[key]
        merged.append(row)

    for index in range(max_len, len(incoming_norm)):
        incoming_row = incoming_norm[index]
        if not _is_kpi_row_empty(incoming_row):
            merged.append(dict(incoming_row))

    return _sanitize_kpi_list(merged)


def _message_mentions_kpi_instruction(message: str) -> bool:
    normalized = _normalize_text_for_match(message)
    return any(marker in normalized for marker in _KPI_INSTRUCTION_MARKERS)


def _is_meta_instruction_message(message: str) -> bool:
    if _is_conversational_feedback_message(message):
        return True
    normalized = _normalize_text_for_match(message)
    if not normalized:
        return False
    if any(verb in normalized for verb in _INSTRUCTION_VERBS):
        if any(
            token in normalized
            for token in ("kpi", "kpis", "indicador", "base", "meta", "linea base")
        ):
            return True
        if normalized.startswith(("agrega", "agregar", "anade", "añade", "pon ", "ponle", "modifica", "actualiza")):
            return True
    if any(token in normalized for token in ("ponle", "pongas", "le pongas")):
        return True
    return False


def _extract_kpi_value_for_target(sub: str, message: str) -> Optional[str]:
    raw = (message or "").strip()
    if not raw or _is_affirmative_message(raw):
        return None

    lowered = raw.lower()
    prefixes = {
        "indicador": ("indicador", "metrica", "métrica", "kpi"),
        "base": ("linea base", "línea base", "base"),
        "meta": ("meta", "objetivo", "valor deseado"),
    }
    for prefix in prefixes.get(sub, ()):
        marker = f"{prefix}:"
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            candidate = raw[start:].strip(" .:-")
            if candidate:
                return candidate
        marker = f"{prefix} "
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            candidate = raw[start:].strip(" .:-")
            if candidate:
                return candidate

    if _is_meta_instruction_message(raw) and not any(char.isdigit() for char in raw):
        return None
    if len(raw) > 200:
        return None
    if "?" in raw and _message_mentions_kpi_instruction(raw):
        return None
    return raw


def _filter_guided_form_updates(
    form_updates: List[Dict[str, Any]],
    write_target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]],
    merged_before: dict,
) -> List[Dict[str, Any]]:
    if not write_target:
        return form_updates

    filtered: List[Dict[str, Any]] = []
    for update in form_updates or []:
        name = (update.get("function") or update.get("name") or "").strip()
        args = update.get("args") or {}
        if name == "UpdateFormField":
            field_name = args.get("field_name")
            value = args.get("value")
            if field_name == "kpis":
                continue
            if _is_generic_confirmation_value(value) or _is_meta_instruction_message(str(value or "")):
                continue
            if _is_conversational_feedback_message(str(value or "")):
                continue
            if write_target[0] == "kpi":
                continue
            allowed = {write_target[1], "datos_ubicacion"}
            if field_name not in allowed:
                continue
        elif name == "UpdateKpis":
            if write_target[0] != "kpi":
                continue
            raw_items = args.get("items") if args.get("items") is not None else args.get("kpis")
            if raw_items:
                args["items"] = _merge_kpi_items(merged_before.get("kpis") or [], raw_items)
        elif name == "UpdateBibliography":
            if write_target[0] == "scalar" and write_target[1] != "bibliografia":
                continue
        filtered.append(update)
    return filtered


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
                if field_name == "beneficio_esperado":
                    val = _merge_beneficio_field(base.get(field_name), val)
                elif field_name == "valor_estimado":
                    val = _merge_money_field(base.get(field_name), val)
                base[field_name] = val
        elif name == "UpdateKpis":
            raw_items = args.get("items") if args.get("items") is not None else args.get("kpis")
            if not raw_items:
                continue
            base["kpis"] = _merge_kpi_items(base.get("kpis") or [], raw_items)
        elif name == "UpdateBibliography":
            raw_items = args.get("items") or []
            base["bibliografia"] = _normalize_bibliography_items(raw_items)
    base["kpis"] = _sanitize_kpi_list(base.get("kpis") or [])
    return base


def _next_guided_target(merged: dict) -> Optional[Union[Tuple[str, str], Tuple[str, int, str]]]:
    for key in _GUIDED_SCALAR_ORDER:
        if key == "beneficio_esperado":
            beneficio_target = _beneficio_field_target(merged.get(key))
            if beneficio_target:
                return beneficio_target
            continue
        if key == "valor_estimado":
            money_target = _money_field_target(key, merged.get(key))
            if money_target:
                return money_target
            continue
        if _is_empty_field(key, merged.get(key)):
            return ("scalar", key)
    kpis = _sanitize_kpi_list(merged.get("kpis") or [])
    if not kpis:
        return ("kpi", 0, "indicador")
    for index, row in enumerate(kpis):
        if not isinstance(row, dict):
            row = {}
        if _is_kpi_row_empty(row):
            if len(kpis) == 1:
                return ("kpi", index, "indicador")
            continue
        for sub in ("indicador", "base", "meta"):
            if _is_empty_field(sub, row.get(sub)):
                return ("kpi", index, sub)
    if _bibliography_needs_fill(merged):
        return ("scalar", "bibliografia")
    return None


def _guided_question(target: Optional[Union[Tuple[str, str], Tuple[str, int, str]]]) -> str:
    if target is None:
        return (
            "Con esto tenemos un borrador completo con bibliografía incluida. "
            "Revisa el formulario a la izquierda; si quieres, dime qué ajustar."
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
            "beneficio_esperado": "Dime el beneficio esperado y un valor monetario estimado para usarlo en el ROI.",
            "valor_estimado": "Dime el costo estimado de implementar la iniciativa; si no tienes cifra, te ayudo a estimarlo.",
            "bibliografia": (
                "Para sustentar la iniciativa necesitamos al menos 3 fuentes confiables. "
                "Te propongo algunas URLs con su uso; dime si las agregamos al formulario "
                "o si prefieres otras."
            ),
        }
        return q.get(
            key,
            "Sigamos con el siguiente dato. ¿Puedes completar este campo en una frase?",
        )
    if kind == "money":
        _kind, field_name, sub = target
        if field_name == "valor_estimado":
            if sub == "texto":
                return (
                    "Ahora definamos el costo estimado. Describe qué incluye el costo "
                    "de implementación: desarrollo, equipos, integraciones, infraestructura o soporte."
                )
            return "Indícame el monto del costo estimado en dólares. Ejemplo: 2000."
        if field_name == "beneficio_esperado":
            if sub == "texto":
                return (
                    "Describe el beneficio esperado: qué valor generará la iniciativa "
                    "para la operación o el negocio."
                )
            if sub == "comunicacion":
                return (
                    "¿Qué comunicación facilita entre equipos o áreas y cómo "
                    "(reportes, alertas, dashboards, notificaciones...)?"
                )
            return "Indícame el valor monetario estimado del beneficio en dólares. Ejemplo: 20400."
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
    tools = [UpdateFormField, UpdateKpis, UpdateBibliography]
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
                "kpis", "beneficio_esperado", "valor_estimado", "bibliografia",
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
- Si el usuario corrige, conversa o pide cambios ("no, me parece que...", "ponle que..."), no copies literal su mensaje al formulario: interpreta, reformula en lenguaje de negocio y recién entonces usa la herramienta.
- Si el usuario no sabe, propone 1-2 opciones concretas. Puedes prellenar un borrador con tools cuando la idea dé evidencia suficiente, pero deja claro que es editable.
- Si propones un valor para un campo, menciona el nombre del campo en lenguaje natural para que el usuario sepa qué está aprobando.
- Antes de pedir cada campo, explica en una frase qué significa. Ejemplo: "Resultado esperado significa qué debería mejorar si esto funciona".
- Explica MVP así: "una primera versión simple para probar si la idea funciona, sin construir todo desde el inicio".
- Para datos necesarios, pregunta en lenguaje simple: qué información usa hoy el equipo, dónde vive (Excel, sistema, correos, PDFs, reportes) y quién la tiene.
- Explica validación como: "cómo sabremos que la iniciativa funcionó".
- Después de validación, pide primero el costo estimado en dos pasos: 1) qué incluye el costo, 2) monto del costo estimado.
- Luego pide el beneficio esperado en tres pasos: 1) descripción del beneficio, 2) comunicación que facilita, 3) valor monetario estimado.
- Para beneficio_esperado, guarda internamente {'texto': 'beneficio cualitativo', 'comunicacion': '...', 'valor': numero}. Si solo tienes parte de la información, conserva lo anterior y completa la parte faltante. Nunca muestres JSON ni diccionarios crudos al usuario; describe los datos en lenguaje natural con viñetas.
- Para valor_estimado, interpreta y guarda el costo estimado de implementar la iniciativa como {'texto': 'supuestos/costo', 'valor': numero}; no lo uses como beneficio. Si solo tienes texto o solo monto, conserva lo anterior y completa la parte faltante. Nunca muestres JSON ni diccionarios crudos al usuario.
- Convierte beneficios vagos en KPIs sugeridos con indicador, línea base y meta.
- Tras completar los KPIs, solicita bibliografía: propone al menos 3 URLs confiables con su uso (`uso`) y, si el usuario confirma, usa **UpdateBibliography** con items [{url: 'https://...', uso: '...'}, ...].
- Si el usuario dice "agréguelas", "agregalas" o confirma las URLs propuestas, guarda esas URLs con UpdateBibliography; no digas que actualizaste si no llamaste la herramienta.
- Si el usuario no sabe el costo estimado, ayuda a estimarlo con horas, personas involucradas, integraciones, infraestructura o soporte. Si no alcanza, propón dejarlo como estimación preliminar editable.
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
3.1. Para 'valor_estimado', devuelve {{'texto': '...', 'valor': numero}}; este campo representa costo estimado de implementación, no beneficio. Si el usuario solo da el monto, devuelve el valor numérico en 'valor' y no inventes otro texto.
3.2. Para 'beneficio_esperado', devuelve {{'texto': '...', 'comunicacion': '...', 'valor': numero}} cuando puedas estimar un beneficio monetario. Si el usuario solo da el monto, devuelve el valor numérico en 'valor'.
4. **KPIs (indicadores)**: No uses UpdateFormField para los KPIs. Usa la herramienta **'UpdateKpis'** con el array **'items'**: cada elemento debe incluir **indicador**, **base** y **meta** (strings). Si el usuario pide completar base o meta de un KPI existente, **actualiza esa fila** en el array; no crees filas nuevas vacías. Si añade otro KPI, conserva los anteriores con sus datos y agrega solo la fila nueva.
5. Si el usuario menciona varios KPIs en un mensaje, consolídalos en un solo UpdateKpis.
5.1. Para bibliografía, usa **UpdateBibliography** con items [{url: 'https://...', uso: 'para qué sirve'}, ...]. Nunca uses UpdateFormField para bibliografia. Si propones URLs en el chat y el usuario confirma, debes llamar UpdateBibliography con esas URLs y un uso breve por fuente.
6. No copies al formulario instrucciones meta del chat (por ejemplo "agrega la base", "pon otro KPI") ni el texto literal del usuario si está corrigiendo o conversando; interpreta la intención, reformula y actualiza el campo correcto.
7. Si estás en modo guiado y el formulario aún está vacío, trata el primer mensaje del usuario como la idea inicial; no le pidas un título antes de entender esa idea.

CAMPOS TÉCNICOS DISPONIBLES (field_name en UpdateFormField, exactamente así):
- 'titulo': Nombre de la iniciativa.
- 'unidad': Unidad de negocio (debe ser estrictamente una de estas opciones: 'Aldea Global', 'Aldea Coffee', 'AldeaZON', 'Fundación Aldea', 'Contabilidad', 'Talento Humano', 'Certificación', 'TI', 'Productos').
- 'problema_oportunidad': El problema o la oportunidad.
- 'resultado_esperado': Qué se espera lograr.
- 'mvp': Producto Mínimo Viable.
- 'datos_necesarios': Qué datos se requieren.
- 'datos_ubicacion': Dónde están los datos.
- 'impacto_operacion': Qué cambiará en la operación.
- 'validacion_exito': Cómo se medirá el éxito.
- 'beneficio_esperado': Beneficio esperado con texto y valor monetario estimado.
- 'valor_estimado': Costo estimado de la iniciativa con texto, periodo y supuestos clave.
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
    confirmation_target = None
    response = llm_with_tools.invoke(messages)

    form_updates: List[Dict[str, Any]] = []
    for call in _normalize_tool_calls(response):
        form_updates.append(
            {
                "function": call["name"],
                "args": call.get("args") or {},
            }
        )

    for update in form_updates:
        if update.get("function") != "UpdateKpis":
            continue
        args = update.get("args") or {}
        raw_items = args.get("items") if args.get("items") is not None else args.get("kpis")
        if raw_items:
            args["items"] = _merge_kpi_items(merged_before.get("kpis") or [], raw_items)

    if guided_mode:
        confirmation_target = _resolve_guided_confirmation_target(
            history,
            merged_before,
            target_before,
            message,
        )
        write_target = (
            confirmation_target
            if _is_affirmative_message(message)
            else target_before
        )
        form_updates = _filter_guided_form_updates(form_updates, write_target, merged_before)
        if _is_conversational_feedback_message(message) and not _is_affirmative_message(message):
            unit_override = _extract_business_unit_from_message(message)
            correcting_unidad = _is_unit_correction_context(history, target_before)
            form_updates = [
                update
                for update in form_updates
                if update.get("function") != "UpdateFormField"
                or (
                    correcting_unidad
                    and unit_override
                    and (update.get("args") or {}).get("field_name") == "unidad"
                    and _normalize_business_unit((update.get("args") or {}).get("value")) == unit_override
                )
            ]

    unit_correction_applied = False
    if guided_mode:
        unit_correction_applied = _append_guided_unit_correction_fallback(
            form_updates,
            history=history,
            target_before=target_before,
            message=message,
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
        if inferred:
            args["value"] = inferred

    if guided_mode:
        _append_guided_confirmation_fallback(
            form_updates,
            history=history,
            merged_before=merged_before,
            target=confirmation_target,
            message=message,
        )
        _append_guided_numeric_fallback(
            form_updates,
            target=target_before,
            message=message,
        )

    _append_bibliography_confirmation_fallback(
        form_updates,
        history=history,
        message=message,
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

    money_state = _apply_form_updates(current_form, [])
    for update in form_updates:
        if update.get("function") != "UpdateFormField":
            continue
        args = update.get("args") or {}
        field_name = args.get("field_name")
        if field_name == "beneficio_esperado":
            args["value"] = _merge_beneficio_field(money_state.get(field_name), args.get("value"))
            money_state[field_name] = args["value"]
        elif field_name == "valor_estimado":
            args["value"] = _merge_money_field(money_state.get(field_name), args.get("value"))
            money_state[field_name] = args["value"]

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

    form_updates = _strip_invalid_guided_form_updates(
        form_updates,
        message,
        guided_mode=guided_mode,
    )

    content_out = _append_guided_followup(
        response.content or "", guided_mode, current_form, form_updates
    )
    if unit_correction_applied:
        corrected_content = _build_unit_correction_content(current_form, form_updates)
        if corrected_content:
            content_out = corrected_content
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
    if guided_mode and _is_generic_update_reply(content_out) and not unit_correction_applied:
        merged_after = _apply_form_updates(current_form, form_updates)
        nxt_after = _next_guided_target(merged_after)
        if nxt_after is not None:
            forced = _guided_specific_proposal(nxt_after, merged_after) or _guided_question(nxt_after)
            content_out = f"Listo, avanzamos. {forced}"

    content_out = _humanize_chat_response(content_out)

    return {
        "content": content_out,
        "form_updates": form_updates,
    }

