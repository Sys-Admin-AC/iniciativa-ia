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
from app.roi import normalize_roi_form_fields

router = APIRouter(tags=["history"])


def _json_loads(value, default):
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed


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
        .outerjoin(db.InitiativeWorkflow, db.Conversation.id == db.InitiativeWorkflow.conversation_id)
        .filter(db.Conversation.user_id == user_id)
        .filter(
            (db.InitiativeWorkflow.id == None) | (db.InitiativeWorkflow.current_status == "draft")
        )
        .order_by(desc(last_activity))
        .all()
    )
    return [
        {
            "id": c.id,
            "initiative_title": c.initiative_title,
            "form_data": c.form_data,
            "potenciadores": _json_loads(c.potenciadores, None),
            "roi_detalle": _json_loads(c.roi_detalle, None),
            "roi": c.roi,
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

    form_data = _json_loads(conv.form_data, {})
    if isinstance(form_data, dict):
        form_data = normalize_roi_form_fields(form_data)
    potenciadores = _json_loads(conv.potenciadores, None)
    roi_detalle = _json_loads(conv.roi_detalle, None)
    if isinstance(form_data, dict) and potenciadores:
        form_data["potenciadores"] = potenciadores

    return {
        "id": conv.id,
        "title": conv.initiative_title,
        "analysis": analysis,
        "chat_history": chat_history,
        "form_data": form_data,
        "potenciadores": potenciadores,
        "roi_detalle": roi_detalle,
        "roi": conv.roi,
        "created_at": _iso(conv.created_at),
    }


@router.delete("/history/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    conv = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == conversation_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conv.user_id is not None and conv.user_id != user_id:
        raise HTTPException(status_code=403, detail="No tiene permiso para eliminar esta conversación")

    if conv.workflow and conv.workflow.current_status != "draft":
        raise HTTPException(status_code=400, detail="No se puede eliminar una iniciativa en revisión")

    if conv.workflow:
        session.query(db.InitiativeTechnicalEvaluation).filter(
            db.InitiativeTechnicalEvaluation.workflow_id == conv.workflow.id
        ).delete()
        session.query(db.InitiativeTimelineEvent).filter(
            db.InitiativeTimelineEvent.workflow_id == conv.workflow.id
        ).delete()
        session.delete(conv.workflow)

    session.query(db.Message).filter(db.Message.conversation_id == conversation_id).delete()
    session.delete(conv)
    session.commit()
    return {"status": "ok"}
