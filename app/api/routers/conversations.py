import json
import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import agent_logic, db
from app.api.deps import _ensure_conversation_owner, _require_user_id
from app.schemas.initiative import ChatInput, InitiativeInput

router = APIRouter(tags=["conversations"])


@router.post("/conversations")
def create_conversation(
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    """
    Crea una conversación vacía (sin análisis de IA) para chatear sin duplicar
    filas de 'Nueva Iniciativa' que antes se generaban al llamar a /analyze desde ensureConversation.
    """
    conv_id = str(uuid.uuid4())
    new_conv = db.Conversation(
        id=conv_id,
        initiative_title="Nueva Iniciativa",
        form_data=None,
        user_id=user_id,
    )
    session.add(new_conv)
    session.commit()
    return {"conversation_id": conv_id}


@router.post("/analyze")
def analyze(
    data: InitiativeInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    form_payload = data.model_dump(exclude={"conversation_id"})

    try:
        analysis = agent_logic.analyze_initiative(form_payload)
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with AI service.")

    title = data.titulo or "Nueva Iniciativa"
    form_json = json.dumps(form_payload)

    if data.conversation_id:
        existing = (
            session.query(db.Conversation)
            .filter(db.Conversation.id == data.conversation_id)
            .first()
        )
        if existing:
            _ensure_conversation_owner(existing, user_id)
            existing.initiative_title = title
            existing.form_data = form_json
            session.commit()

            messages = (
                session.query(db.Message)
                .filter(db.Message.conversation_id == data.conversation_id)
                .order_by(db.Message.created_at.asc())
                .all()
            )
            if messages:
                if messages[0].role == "user":
                    first_ts = messages[0].created_at or datetime.utcnow()
                    session.add(
                        db.Message(
                            conversation_id=data.conversation_id,
                            role="agent",
                            content=analysis,
                            created_at=first_ts - timedelta(milliseconds=1),
                        )
                    )
                else:
                    messages[0].content = analysis
            else:
                session.add(
                    db.Message(
                        conversation_id=data.conversation_id,
                        role="agent",
                        content=analysis,
                    )
                )
            session.commit()
            return {"conversation_id": data.conversation_id, "analysis": analysis}

    conv_id = str(uuid.uuid4())
    new_conv = db.Conversation(
        id=conv_id, initiative_title=title, form_data=form_json, user_id=user_id
    )
    session.add(new_conv)
    session.commit()

    new_msg = db.Message(conversation_id=conv_id, role="agent", content=analysis)
    session.add(new_msg)
    session.commit()

    return {"conversation_id": conv_id, "analysis": analysis}


@router.post("/chat")
def chat(
    data: ChatInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    conv = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == data.conversation_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _ensure_conversation_owner(conv, user_id)

    history_records = (
        session.query(db.Message)
        .filter(db.Message.conversation_id == data.conversation_id)
        .order_by(db.Message.created_at.asc())
        .all()
    )
    history = [{"role": msg.role, "content": msg.content} for msg in history_records]

    user_msg = db.Message(
        conversation_id=data.conversation_id, role="user", content=data.message
    )
    session.add(user_msg)

    if data.current_form:
        conv.form_data = json.dumps(data.current_form)
        if data.current_form.get("titulo"):
            conv.initiative_title = data.current_form.get("titulo")

    session.commit()

    try:
        response_data = agent_logic.chat_with_agent(
            data.message,
            history,
            current_form=data.current_form,
            guided_mode=bool(data.guided_mode),
        )
        response_text = response_data["content"]
        form_updates = response_data["form_updates"]
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with AI service.")

    agent_msg = db.Message(
        conversation_id=data.conversation_id, role="agent", content=response_text
    )
    session.add(agent_msg)
    session.commit()

    return {"response": response_text, "form_updates": form_updates}
