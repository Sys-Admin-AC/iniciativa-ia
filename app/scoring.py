from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


SCORE_VERSION = "2026-05"

_POSITIVE_WEIGHTS = {
    "financiero": 0.25,
    "estrategico": 0.20,
    "cliente": 0.15,
    "datos_ia": 0.15,
    "time_to_value": 0.10,
}
_POSITIVE_WEIGHT_TOTAL = sum(_POSITIVE_WEIGHTS.values())

BUSINESS_SCORE_CATEGORIES = {
    "impacto": {
        "financiero": {
            "peso": round(_POSITIVE_WEIGHTS["financiero"] / _POSITIVE_WEIGHT_TOTAL, 4),
            "direction": "positive",
        },
        "estrategico": {
            "peso": round(_POSITIVE_WEIGHTS["estrategico"] / _POSITIVE_WEIGHT_TOTAL, 4),
            "direction": "positive",
        },
        "cliente": {
            "peso": round(_POSITIVE_WEIGHTS["cliente"] / _POSITIVE_WEIGHT_TOTAL, 4),
            "direction": "positive",
        },
    },
    "agilidad": {
        "datos_ia": {
            "peso": round(_POSITIVE_WEIGHTS["datos_ia"] / _POSITIVE_WEIGHT_TOTAL, 4),
            "direction": "positive",
        },
        "time_to_value": {
            "peso": round(_POSITIVE_WEIGHTS["time_to_value"] / _POSITIVE_WEIGHT_TOTAL, 4),
            "direction": "positive",
        },
    },
}

TECHNICAL_CRITERION_KEYS = ("complejidad", "riesgo")

BUSINESS_CRITERION_KEYS = tuple(
    criterion
    for criteria in BUSINESS_SCORE_CATEGORIES.values()
    for criterion in criteria.keys()
)


def default_business_scores(default_score: int = 3) -> Dict[str, int]:
    return {criterion: default_score for criterion in BUSINESS_CRITERION_KEYS}


def _coerce_score(value: Any, default: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(5, score))


def normalize_business_scores(scores: Mapping[str, Any]) -> Dict[str, int]:
    return {
        criterion: _coerce_score(scores.get(criterion))
        for criterion in BUSINESS_CRITERION_KEYS
    }


def criteria_to_scores(criteria: Iterable[Any]) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for item in criteria:
        if isinstance(item, Mapping):
            criterion = item.get("criterion")
            score = item.get("score")
        else:
            criterion = getattr(item, "criterion", None)
            score = getattr(item, "score", None)
        if criterion in BUSINESS_CRITERION_KEYS:
            scores[str(criterion)] = _coerce_score(score)
    return normalize_business_scores(scores)


def _short_text(value: Any, max_length: int = 150) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def normalize_business_explanations(
    explanations: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    source = explanations or {}
    return {
        criterion: _short_text(source.get(criterion))
        for criterion in BUSINESS_CRITERION_KEYS
    }


def criteria_to_explanations(criteria: Iterable[Any]) -> Dict[str, str]:
    explanations: Dict[str, str] = {}
    for item in criteria:
        if isinstance(item, Mapping):
            criterion = item.get("criterion")
            comment = item.get("comment") or item.get("explicacion")
        else:
            criterion = getattr(item, "criterion", None)
            comment = getattr(item, "comment", None) or getattr(item, "explicacion", None)
        if criterion in BUSINESS_CRITERION_KEYS:
            explanations[str(criterion)] = _short_text(comment)
    return normalize_business_explanations(explanations)


def build_potenciadores_payload(
    scores: Mapping[str, Any],
    *,
    estado: str,
    fuente: str,
    comentario: Optional[str] = "",
    explanations: Optional[Mapping[str, Any]] = None,
    resumen: Optional[str] = None,
    actualizado_por: Optional[str] = None,
    actualizado_en: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_scores = normalize_business_scores(scores)
    normalized_explanations = normalize_business_explanations(explanations)
    payload: Dict[str, Any] = {
        "version": SCORE_VERSION,
        "score": 0.0,
        "estado": estado,
        "fuente": fuente,
    }
    total = 0.0

    for category, criteria in BUSINESS_SCORE_CATEGORIES.items():
        payload[category] = {}
        for criterion, meta in criteria.items():
            weight = float(meta["peso"])
            score = normalized_scores[criterion]
            contribution = score * weight
            if meta["direction"] == "negative":
                contribution *= -1
            contribution = round(contribution, 2)
            total += contribution
            payload[category][criterion] = {
                "peso": weight,
                "puntaje": score,
                "aporte": contribution,
                "explicacion": normalized_explanations.get(criterion, ""),
            }

    payload["score"] = round(total, 2)
    payload["resumen"] = _short_text(resumen or comentario)
    payload["comentario"] = comentario or ""
    payload["actualizado_por"] = actualizado_por
    payload["actualizado_en"] = actualizado_en
    return payload


def time_to_value_cap_for_days(days: Optional[int]) -> Optional[int]:
    if days is None:
        return None
    if days <= 90:
        return None
    if days <= 180:
        return 3
    return 2


def apply_time_to_value_from_days(
    score: int,
    days: Optional[int],
    explanation: str = "",
) -> Tuple[int, str]:
    cap = time_to_value_cap_for_days(days)
    if cap is None:
        return score, explanation
    adjusted = min(score, cap)
    suffix = (
        f" Ajustado por plazo de implementación ({days} días): "
        f"{'≤90 días sin penalización' if days <= 90 else 'máximo permitido ' + str(cap)}."
    )
    if adjusted < score:
        note = f"Penalizado por plazo >90 días ({days} días).{suffix}"
        merged = f"{explanation} {note}".strip() if explanation else note
        return adjusted, merged
    if days and days > 90:
        note = f"Plazo de implementación: {days} días (tope {cap} para time_to_value)."
        merged = f"{explanation} {note}".strip() if explanation else note
        return adjusted, merged
    return adjusted, explanation


def apply_implementation_days_to_potenciadores(
    potenciadores: Dict[str, Any],
    days: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(potenciadores, dict) or days is None:
        return potenciadores
    payload = dict(potenciadores)
    agilidad = dict(payload.get("agilidad") or {})
    ttv = dict(agilidad.get("time_to_value") or {})
    current_score = _coerce_score(ttv.get("puntaje"))
    explanation = str(ttv.get("explicacion") or "")
    adjusted_score, adjusted_explanation = apply_time_to_value_from_days(
        current_score,
        days,
        explanation,
    )
    if adjusted_score == current_score and adjusted_explanation == explanation:
        return potenciadores
    ttv["puntaje"] = adjusted_score
    ttv["explicacion"] = adjusted_explanation
    weight = float(ttv.get("peso") or BUSINESS_SCORE_CATEGORIES["agilidad"]["time_to_value"]["peso"])
    contribution = round(adjusted_score * weight, 2)
    ttv["aporte"] = contribution
    agilidad["time_to_value"] = ttv
    payload["agilidad"] = agilidad
    total = 0.0
    for category, criteria in BUSINESS_SCORE_CATEGORIES.items():
        block = payload.get(category) or {}
        for criterion in criteria:
            item = block.get(criterion) or {}
            total += float(item.get("aporte") or 0)
    payload["score"] = round(total, 2)
    return payload
