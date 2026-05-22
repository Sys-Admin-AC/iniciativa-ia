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


def normalize_roi_form_fields(form_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(form_payload or {})
    payload["beneficio_esperado"] = normalize_money_field(payload.get("beneficio_esperado"))
    payload["valor_estimado"] = normalize_money_field(payload.get("valor_estimado"))
    return payload


def money_field_text(value: Any) -> str:
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

    beneficio = normalize_money_field(payload.get("beneficio_esperado")).get("valor")
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
