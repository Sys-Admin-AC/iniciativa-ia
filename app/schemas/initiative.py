from typing import Any, List, Optional

from pydantic import BaseModel, field_validator

from app.roi import normalize_money_field


class KpiInput(BaseModel):
    indicador: str
    base: str
    meta: str


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


class ChatInput(BaseModel):
    conversation_id: str
    message: str
    current_form: Optional[dict] = None
    guided_mode: bool = False
