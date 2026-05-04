import json
from typing import Set

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app import db
from app.api.deps import (
    _ensure_conversation_read_access,
    _get_rbac_roles,
    _iso,
    _require_user_id,
)

router = APIRouter(tags=["history"])


@router.get("/history")
def get_history(
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    """Ordena por última actividad (último mensaje); si aún no hay mensajes, por fecha de creación.
    Solo listado de conversaciones del usuario autenticado (X-User-Id)."""
    subq = (
        session.query(
            db.Message.conversation_id,
            func.max(db.Message.created_at).label("last_at"),
        )
        .group_by(db.Message.conversation_id)
        .subquery()
    )
    last_activity = func.coalesce(subq.c.last_at, db.Conversation.created_at)
    rows = (
        session.query(db.Conversation, last_activity.label("last_at"))
        .outerjoin(subq, db.Conversation.id == subq.c.conversation_id)
        .filter(db.Conversation.user_id == user_id)
        .order_by(desc(last_activity))
        .all()
    )
    return [
        {
            "id": c.id,
            "initiative_title": c.initiative_title,
            "form_data": c.form_data,
            "created_at": _iso(c.created_at),
            "last_activity_at": _iso(la),
        }
        for c, la in rows
    ]


@router.get("/history/{conversation_id}")
def get_conversation_detail(
    conversation_id: str,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    conv = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == conversation_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _ensure_conversation_read_access(conv, user_id, roles)
    session.commit()

    messages = (
        session.query(db.Message)
        .filter(db.Message.conversation_id == conversation_id)
        .order_by(db.Message.created_at.asc())
        .all()
    )

    analysis = ""
    chat_history = []

    if messages:
        if messages[0].role == "agent":
            analysis = messages[0].content
            chat_history = [{"role": m.role, "content": m.content} for m in messages[1:]]
        else:
            analysis = ""
            chat_history = [{"role": m.role, "content": m.content} for m in messages]

    form_data = {}
    if conv.form_data:
        try:
            form_data = json.loads(conv.form_data)
        except json.JSONDecodeError:
            form_data = {}

    return {
        "id": conv.id,
        "title": conv.initiative_title,
        "analysis": analysis,
        "chat_history": chat_history,
        "form_data": form_data,
        "created_at": _iso(conv.created_at),
    }
