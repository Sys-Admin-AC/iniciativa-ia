from typing import Any, Dict, Iterable, Mapping, Optional


SCORE_VERSION = "2026-05"

BUSINESS_SCORE_CATEGORIES = {
    "impacto": {
        "financiero": {"peso": 0.25, "direction": "positive"},
        "estrategico": {"peso": 0.20, "direction": "positive"},
        "cliente": {"peso": 0.15, "direction": "positive"},
    },
    "agilidad": {
        "datos_ia": {"peso": 0.15, "direction": "positive"},
        "time_to_value": {"peso": 0.10, "direction": "positive"},
    },
    "obstaculos": {
        "complejidad": {"peso": 0.08, "direction": "negative"},
        "riesgo": {"peso": 0.07, "direction": "negative"},
    },
}

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
    return max(1, min(5, score))


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
