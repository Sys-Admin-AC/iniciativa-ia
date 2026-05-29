import re
from typing import Any, Dict, Optional


ROI_VERSION = "2026-05"


def _parse_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value).strip().lower()
    if not text:
        return None

    multiplier = 1.0
    if re.search(r"\b(mill[oó]n|millones|mm)\b", text):
        multiplier = 1_000_000.0
    elif re.search(r"\b(mil|k)\b", text):
        multiplier = 1_000.0

    match = re.search(r"\d+(?:[.,]\d+)*(?:[.,]\d+)?", text)
    if not match:
        return None

    raw = match.group(0)
    if "," in raw and "." in raw:
        last_dot = raw.rfind(".")
        last_comma = raw.rfind(",")
        if last_comma > last_dot:
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        parts = raw.split(",")
        raw = raw.replace(",", "") if len(parts[-1]) == 3 else raw.replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")

    try:
        return round(float(raw) * multiplier, 2)
    except ValueError:
        return None


def normalize_money_field(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        text = str(
            value.get("texto")
            or value.get("text")
            or value.get("descripcion")
            or value.get("description")
            or ""
        ).strip()
        numeric = _parse_number(
            value.get("valor")
            if value.get("valor") is not None
            else value.get("value")
            if value.get("value") is not None
            else value.get("monto")
            if value.get("monto") is not None
            else value.get("cantidad")
        )
        if numeric is None:
            numeric = _parse_number(text)
        return {"texto": text, "valor": numeric}

    text = "" if value is None else str(value).strip()
    return {"texto": text, "valor": _parse_number(value)}


def normalize_beneficio_field(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        legacy = normalize_money_field(value)
        return {
            "texto": legacy.get("texto") or "",
            "comunicacion": "",
            "valor": legacy.get("valor"),
        }

    legacy_text = str(value.get("texto") or "").strip()
    if not legacy_text:
        legacy_text = str(
            value.get("beneficio_familia_aldea")
            or value.get("impacto_operativo")
            or ""
        ).strip()

    comunicacion_raw = value.get("comunicacion")
    if isinstance(comunicacion_raw, dict):
        parts = [
            str(comunicacion_raw.get("facilita") or "").strip(),
            str(comunicacion_raw.get("como") or "").strip(),
        ]
        comunicacion = ". ".join(part for part in parts if part)
    else:
        comunicacion = str(comunicacion_raw or "").strip()

    return {
        "texto": legacy_text,
        "comunicacion": comunicacion,
        "valor": normalize_money_field({"valor": value.get("valor")}).get("valor"),
    }


def normalize_implementation_days(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        days = int(float(value))
    except (TypeError, ValueError):
        return None
    if days < 1:
        return None
    return min(days, 730)


def estimate_implementation_days(form_payload: Dict[str, Any]) -> int:
    payload = form_payload or {}
    mvp = str(payload.get("mvp") or "").lower()
    datos = str(payload.get("datos_necesarios") or "").lower()
    days = 75
    complexity_markers = (
        "integracion", "integración", "erp", "sap", "legacy", "migracion", "migración",
        "multi", "varios sistemas", "tiempo real", "produccion", "producción",
    )
    for marker in complexity_markers:
        if marker in mvp or marker in datos:
            days += 15
    if any(token in mvp for token in ("ia", "machine learning", "modelo", "llm")):
        days += 10
    if len(mvp) > 240:
        days += 20
    return max(30, min(730, days))


def effective_implementation_days(form_payload: Dict[str, Any]) -> Optional[int]:
    payload = form_payload or {}
    ti_days = normalize_implementation_days(payload.get("dias_implementacion_ti"))
    if ti_days is not None:
        return ti_days
    return normalize_implementation_days(payload.get("dias_implementacion_estimados"))


def beneficio_field_text(value: Any) -> str:
    normalized = normalize_beneficio_field(value)
    parts = []
    if normalized.get("texto"):
        parts.append(str(normalized["texto"]))
    if normalized.get("comunicacion"):
        parts.append(f"Comunicación: {normalized['comunicacion']}")
    amount = normalized.get("valor")
    if amount is not None:
        parts.append(f"Valor estimado del beneficio: {amount:g}")
    return "\n".join(parts) if parts else "Sin descripción"


PERSISTED_ANALYSIS_FORM_FIELDS = (
    "bibliografia_analisis",
    "calidad_datos_analisis",
    "factor_innovador_analisis",
    "resumen_iniciativa",
    "dias_implementacion_estimados",
)


def merge_persisted_analysis_fields(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(incoming or {})
    if not isinstance(existing, dict):
        return merged
    for key in PERSISTED_ANALYSIS_FORM_FIELDS:
        if merged.get(key):
            continue
        value = existing.get(key)
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def normalize_roi_form_fields(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(form_payload or {})
    payload["beneficio_esperado"] = normalize_beneficio_field(payload.get("beneficio_esperado"))
    payload["valor_estimado"] = normalize_money_field(payload.get("valor_estimado"))
    estimated = normalize_implementation_days(payload.get("dias_implementacion_estimados"))
    if estimated is not None:
        payload["dias_implementacion_estimados"] = estimated
    elif not payload.get("dias_implementacion_estimados"):
        payload["dias_implementacion_estimados"] = estimate_implementation_days(payload)
    payload["dias_implementacion_ti"] = normalize_implementation_days(
        payload.get("dias_implementacion_ti")
    )
    innov = payload.get("factor_innovador_ti")
    if isinstance(innov, dict):
        score = innov.get("puntaje")
        if score is not None and score != "":
            try:
                payload["factor_innovador_ti"] = {
                    "puntaje": max(0, min(5, int(score))),
                    "comentario": str(innov.get("comentario") or "").strip(),
                }
            except (TypeError, ValueError):
                payload["factor_innovador_ti"] = {
                    "puntaje": innov.get("puntaje"),
                    "comentario": str(innov.get("comentario") or "").strip(),
                }
    return payload


def money_field_text(value: Any) -> str:
    if isinstance(value, dict) and (
        value.get("comunicacion") is not None
        or value.get("impacto_operativo") is not None
        or value.get("beneficio_familia_aldea") is not None
        or isinstance(value.get("comunicacion"), dict)
    ):
        return beneficio_field_text(value)
    normalized = normalize_money_field(value)
    return normalized["texto"]


def money_field_prompt(value: Any, label: str) -> str:
    normalized = normalize_money_field(value)
    text = normalized["texto"] or "Sin descripción"
    amount = normalized["valor"]
    if amount is None:
        return text
    return f"{text} ({label}: {amount:g})"


def clamp_probability(value: Any) -> float:
    parsed = _parse_number(value)
    if parsed is None:
        return 0.6
    if parsed > 1:
        parsed = parsed / 100
    return round(max(0.0, min(1.0, parsed)), 2)


def merge_persisted_roi(
    roi_detalle: Optional[Dict[str, Any]],
    column_roi: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(roi_detalle, dict):
        return None

    detail_roi = roi_detalle.get("roi")
    if column_roi is None or detail_roi not in (None, 0, 0.0):
        return roi_detalle

    try:
        if float(column_roi) == 0:
            return roi_detalle
    except (TypeError, ValueError):
        return roi_detalle

    merged = dict(roi_detalle)
    merged["roi"] = round(float(column_roi), 2)
    return merged


def estimate_success_probability(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = form_payload or {}
    probability = 0.5

    if str(payload.get("mvp") or "").strip():
        probability += 0.1
    if str(payload.get("datos_necesarios") or "").strip() and str(payload.get("datos_ubicacion") or "").strip():
        probability += 0.1
    if str(payload.get("validacion_exito") or "").strip():
        probability += 0.08
    if isinstance(payload.get("kpis"), list) and len(payload.get("kpis") or []) > 0:
        probability += 0.07

    beneficio = normalize_beneficio_field(payload.get("beneficio_esperado")).get("valor")
    costo = normalize_money_field(payload.get("valor_estimado")).get("valor")
    if beneficio is None or costo is None:
        probability -= 0.1

    probability = round(max(0.35, min(0.9, probability)), 2)
    return {
        "p_exito": probability,
        "explicacion": "Estimación automática basada en claridad del MVP, datos, validación y KPIs.",
    }


def build_roi_payload(
    form_payload: Dict[str, Any],
    probability: Any,
    *,
    fuente: str,
    explicacion: str = "",
) -> Dict[str, Any]:
    normalized = normalize_roi_form_fields(form_payload)
    beneficio = normalized["beneficio_esperado"].get("valor")
    costo = normalized["valor_estimado"].get("valor")
    p_exito = clamp_probability(probability)

    roi = None
    if beneficio is not None and costo not in (None, 0):
        roi = round((beneficio * p_exito) / costo, 2)

    return {
        "version": ROI_VERSION,
        "beneficio": beneficio,
        "p_exito": p_exito,
        "costo_estimado": costo,
        "roi": roi,
        "fuente": fuente,
        "explicacion": explicacion,
    }
