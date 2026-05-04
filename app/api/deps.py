import re
from typing import Optional, Set

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


def _get_rbac_roles(
    x_th_rbac_roles: Optional[str] = Header(None, alias="X-Th-Rbac-Roles")
) -> Set[str]:
    """Roles RBAC enviados por el proxy Yii. Acepta valores separados por coma, pipe o espacios."""
    if not x_th_rbac_roles:
        return set()
    return {
        role.strip().lower()
        for role in re.split(r"[,|\s]+", x_th_rbac_roles)
        if role.strip()
    }


def _ensure_conversation_owner(conv: db.Conversation, user_id: int) -> None:
    """Asigna user_id a conversaciones huérfanas o comprueba que coincida con el usuario actual."""
    if conv.user_id is None:
        conv.user_id = user_id
        return
    if conv.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="No tienes acceso a esta conversación."
        )


def _ensure_conversation_read_access(
    conv: db.Conversation,
    user_id: int,
    roles: Set[str],
    allowed_roles: Optional[Set[str]] = None,
) -> None:
    """Permite lectura al dueño o a roles autorizados por el proxy Yii."""
    allowed_roles = allowed_roles or {"ti", "iniciativa_comite", "admin"}
    if conv.user_id is None:
        conv.user_id = user_id
        return
    if conv.user_id == user_id:
        return
    if roles.intersection({role.lower() for role in allowed_roles}):
        return
    raise HTTPException(
        status_code=403, detail="No tienes acceso de lectura a esta conversación."
    )


def _get_conversation_for_mutation(
    conversation_id: str,
    session,
    user_id: int,
    roles: Set[str],
    allowed_roles: Set[str],
):
    """Carga la conversación si el usuario es dueño o tiene un rol permitido (proxy Yii)."""
    conversation = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _ensure_conversation_read_access(
        conversation, user_id, roles, allowed_roles=allowed_roles
    )
    return conversation


def _iso(dt):
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
