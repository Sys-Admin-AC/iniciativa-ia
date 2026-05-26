from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

from app.roi import normalize_money_field


class KpiInput(BaseModel):
    indicador: str
    base: str
    meta: str


class BibliographyInput(BaseModel):
    url: str


class InitiativeInput(BaseModel):
    titulo: str
    unidad: str
    problema_oportunidad: str
    resultado_esperado: str
    mvp: str
    datos_necesarios: str
    datos_ubicacion: str
    impacto_operacion: str
    validacion_exito: str
    kpis: List[KpiInput]
    bibliografia: List[BibliographyInput]
    beneficio_esperado: Any
    valor_estimado: Optional[Any] = None
    # Si viene seteado, se actualiza esa conversación en lugar de crear otra (evita duplicados en el historial).
    conversation_id: Optional[str] = None

    @field_validator("beneficio_esperado", "valor_estimado", mode="before")
    @classmethod
    def normalize_money_fields(cls, value):
        if value is None or value == "":
            return None
        return normalize_money_field(value)

    @field_validator("bibliografia", mode="before")
    @classmethod
    def normalize_bibliography(cls, value):
        items = value if isinstance(value, list) else []
        normalized = []
        for item in items:
            url = item.get("url") if isinstance(item, dict) else item
            url = str(url or "").strip()
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Todas las fuentes deben ser URLs válidas con http:// o https://.")
            normalized.append({"url": url})
        if len(normalized) < 3:
            raise ValueError("La bibliografía debe incluir al menos 3 URLs válidas.")
        return normalized


class ChatInput(BaseModel):
    conversation_id: str
    message: str
    current_form: Optional[dict] = None
    guided_mode: bool = False
