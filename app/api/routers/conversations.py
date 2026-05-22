import json
import logging
import uuid
from datetime import datetime, timedelta

from app.utils import get_now

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import agent_logic, db
from app.api.deps import _ensure_conversation_owner, _require_user_id
from app.roi import build_roi_payload, estimate_success_probability, normalize_roi_form_fields
from app.scoring import build_potenciadores_payload, default_business_scores
from app.schemas.initiative import ChatInput, InitiativeInput

router = APIRouter(tags=["conversations"])


def _build_score_suggestion(form_payload, analysis):
    try:
        return agent_logic.suggest_business_score(form_payload, analysis)
    except Exception as e:
        logging.error(f"Error generating score suggestion: {e}")
        return build_potenciadores_payload(
            default_business_scores(),
            estado="sugerido",
            fuente="sistema",
            comentario="Sugerencia base generada por defecto al no poder calcular con IA.",
        )


def _build_roi_details(form_payload, analysis):
    try:
        probability = agent_logic.suggest_roi_probability(form_payload, analysis)
    except Exception as e:
        logging.error(f"Error generating ROI probability: {e}")
        probability = estimate_success_probability(form_payload)
    return build_roi_payload(
        form_payload,
        probability.get("p_exito"),
        fuente="ia" if probability else "sistema",
        explicacion=probability.get("explicacion", ""),
    )


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
    form_payload = normalize_roi_form_fields(data.model_dump(exclude={"conversation_id"}))

    try:
        analysis = agent_logic.analyze_initiative(form_payload)
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with AI service.")

    potenciadores = _build_score_suggestion(form_payload, analysis)
    roi_detalle = _build_roi_details(form_payload, analysis)
    analysis = agent_logic.append_score_to_analysis(analysis, potenciadores)
    potenciadores_json = json.dumps(potenciadores, ensure_ascii=False)
    roi_detalle_json = json.dumps(roi_detalle, ensure_ascii=False)
    title = data.titulo or "Nueva Iniciativa"
    form_json = json.dumps(form_payload, ensure_ascii=False)

    if data.conversation_id:
        existing = (
            session.query(db.Conversation)
            .filter(db.Conversation.id == data.conversation_id)
            .first()
        )
        if existing:
            _ensure_conversation_owner(existing, user_id)
            if existing.workflow and existing.workflow.current_status != "draft":
                raise HTTPException(status_code=403, detail="No se puede editar una iniciativa en revisión.")
            existing.initiative_title = title
            existing.form_data = form_json
            existing.potenciadores = potenciadores_json
            existing.roi_detalle = roi_detalle_json
            existing.roi = roi_detalle.get("roi")
            session.commit()

            messages = (
                session.query(db.Message)
                .filter(db.Message.conversation_id == data.conversation_id)
                .order_by(db.Message.created_at.asc())
                .all()
            )
            if messages:
                if messages[0].role == "user":
                    first_ts = messages[0].created_at or get_now()
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
            return {
                "conversation_id": data.conversation_id,
                "analysis": analysis,
                "potenciadores": potenciadores,
                "roi_detalle": roi_detalle,
                "roi": roi_detalle.get("roi"),
            }

    conv_id = str(uuid.uuid4())
    new_conv = db.Conversation(
        id=conv_id,
        initiative_title=title,
        form_data=form_json,
        potenciadores=potenciadores_json,
        roi_detalle=roi_detalle_json,
        roi=roi_detalle.get("roi"),
        user_id=user_id,
    )
    session.add(new_conv)
    session.commit()

    new_msg = db.Message(conversation_id=conv_id, role="agent", content=analysis)
    session.add(new_msg)
    session.commit()

    return {
        "conversation_id": conv_id,
        "analysis": analysis,
        "potenciadores": potenciadores,
        "roi_detalle": roi_detalle,
        "roi": roi_detalle.get("roi"),
    }


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

    current_form = data.current_form
    if data.current_form:
        if conv.workflow and conv.workflow.current_status != "draft":
             raise HTTPException(status_code=403, detail="No se puede editar una iniciativa en revisión.")
        current_form = normalize_roi_form_fields(data.current_form)
        conv.form_data = json.dumps(current_form, ensure_ascii=False)
        if data.current_form.get("titulo"):
            conv.initiative_title = data.current_form.get("titulo")

    session.commit()

    try:
        response_data = agent_logic.chat_with_agent(
            data.message,
            history,
            current_form=current_form,
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
