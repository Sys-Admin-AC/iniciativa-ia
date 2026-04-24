from typing import List, Optional

from pydantic import BaseModel, field_validator


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
    beneficio_esperado: str
    valor_estimado: Optional[float] = None
    # Si viene seteado, se actualiza esa conversación en lugar de crear otra (evita duplicados en el historial).
    conversation_id: Optional[str] = None

    @field_validator("valor_estimado", mode="before")
    @classmethod
    def normalize_valor_estimado(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            clean = value.strip()
            if not clean:
                return None
            clean = clean.replace("$", "").replace(",", "")
            return float(clean)
        return value


class ChatInput(BaseModel):
    conversation_id: str
    message: str
    current_form: Optional[dict] = None
    guided_mode: bool = False
