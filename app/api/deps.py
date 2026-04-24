from typing import Optional

from fastapi import Header, HTTPException

from app import db


def _require_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> int:
    """ID del usuario en Yii (tabla `user`). Inyectado por el proxy PHP, no confiar en el cliente directo en producción."""
    if x_user_id is None or str(x_user_id).strip() == "":
        raise HTTPException(
            status_code=401,
            detail="Se requiere el encabezado X-User-Id (sesión de usuario en la app).",
        )
    try:
        uid = int(str(x_user_id).strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="X-User-Id debe ser un entero.")
    if uid < 1:
        raise HTTPException(status_code=400, detail="X-User-Id inválido.")
    return uid


def _ensure_conversation_owner(conv: db.Conversation, user_id: int) -> None:
    """Asigna user_id a conversaciones huérfanas o comprueba que coincida con el usuario actual."""
    if conv.user_id is None:
        conv.user_id = user_id
        return
    if conv.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="No tienes acceso a esta conversación."
        )


def _iso(dt):
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
