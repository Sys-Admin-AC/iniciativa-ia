import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from app.utils import get_now

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app import db
from app.api.deps import (
    _ensure_conversation_owner,
    _ensure_conversation_read_access,
    _get_conversation_for_mutation,
    _get_rbac_roles,
    _iso,
    _require_user_id,
)
from app.roi import (
    build_roi_payload,
    estimate_success_probability,
    merge_persisted_roi,
    normalize_roi_form_fields,
)
from app.scoring import (
    build_potenciadores_payload,
    criteria_to_explanations,
    criteria_to_scores,
    default_business_scores,
)
from app.schemas.initiative_workflow import (
    BusinessEvaluationInput,
    CommitteeResponseInput,
    TechnicalEvaluationInput,
    TimelineEventResponse,
    WorkflowActionInput,
    WorkflowListItemResponse,
    WorkflowStatusResponse,
)

router = APIRouter(prefix="/workflow", tags=["workflow"])

STATUS_DRAFT = "draft"
STATUS_COMMITTEE_TI = "committee_ti_review"
STATUS_COMMITTEE_OPERATIONS = "committee_operations_review"
STATUS_COMMITTEE_MANAGEMENT = "committee_management_review"
STATUS_BACKLOG = "backlog"
STATUS_ANNULLED = "annulled"

EVENT_SUBMITTED_TO_COMMITTEE = "submitted_to_committee"
EVENT_SUBMITTED_TO_OPERATIONS = "submitted_to_operations_committee"
EVENT_TECHNICAL_COMMENT_SUBMITTED = "technical_comment_submitted"
EVENT_BUSINESS_SCORE_SUBMITTED = "business_score_submitted"
EVENT_COMMITTEE_COMMENT_SUBMITTED = "committee_comment_submitted"
EVENT_RETURNED_WITH_OBSERVATIONS = "returned_with_observations"
EVENT_SUBMITTED_TO_MANAGEMENT = "submitted_to_management_committee"
EVENT_SENT_TO_BACKLOG = "sent_to_backlog"
EVENT_ANNULLED = "initiative_annulled"

ROLE_COMMITTEE_MUTATION = frozenset({"iniciativa_comite"})
ROLE_TI_MUTATION = frozenset({"ti"})
ROLE_ADMIN_MUTATION = frozenset({"admin"})
ROLE_OPERATIONS_COMMITTEE_MUTATION = frozenset(
    {
        "iniciativa_comite_operaciones",
        "comite_operaciones",
        "committee_operations",
        "operaciones",
    }
)
ROLE_MANAGEMENT_COMMITTEE_MUTATION = frozenset(
    {
        "iniciativa_comite_gerencial",
        "comite_gerencial",
        "committee_management",
        "gerencial",
    }
)
ROLE_ANY_COMMITTEE_MUTATION = (
    ROLE_COMMITTEE_MUTATION
    .union(ROLE_OPERATIONS_COMMITTEE_MUTATION)
    .union(ROLE_MANAGEMENT_COMMITTEE_MUTATION)
)

LEGACY_STATUS_MAP = {
    "draft_analyzed": STATUS_DRAFT,
    "committee_review": STATUS_COMMITTEE_TI,
    "technical_review": STATUS_COMMITTEE_TI,
    "committee_with_technical_feedback": STATUS_COMMITTEE_OPERATIONS,
    "user_reviewed": STATUS_COMMITTEE_OPERATIONS,
    "final_committee": STATUS_COMMITTEE_MANAGEMENT,
    "rejected": STATUS_BACKLOG,
}

LEGACY_EQUIVALENTS = {
    STATUS_DRAFT: {"draft_analyzed"},
    STATUS_COMMITTEE_TI: {"committee_review", "technical_review"},
    STATUS_COMMITTEE_OPERATIONS: {
        "committee_with_technical_feedback",
        "user_reviewed",
    },
    STATUS_COMMITTEE_MANAGEMENT: {"final_committee"},
    STATUS_BACKLOG: {"rejected"},
}


def _normalized_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    return LEGACY_STATUS_MAP.get(status, status)


def _status_filter_with_legacy(statuses: List[str]) -> List[str]:
    values = set()
    for status in statuses:
        if not status:
            continue
        normalized = _normalized_status(status)
        values.add(normalized)
        values.update(LEGACY_EQUIVALENTS.get(normalized, set()))
    return sorted(values)


def _json_dumps(value: Optional[Dict[str, Any]]) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _conversation_potenciadores(conversation: Optional[db.Conversation]) -> Optional[Dict[str, Any]]:
    if not conversation:
        return None
    parsed = _json_loads(conversation.potenciadores, None)
    if isinstance(parsed, dict):
        return parsed
    form_data = _json_loads(conversation.form_data, {})
    if isinstance(form_data, dict):
        nested = form_data.get("potenciadores")
        if isinstance(nested, dict):
            return nested
    return None


def _conversation_roi_details(conversation: Optional[db.Conversation]) -> Optional[Dict[str, Any]]:
    if not conversation:
        return None
    parsed = _json_loads(conversation.roi_detalle, None)
    if not isinstance(parsed, dict):
        return None
    return merge_persisted_roi(parsed, conversation.roi)


def _ensure_roi_for_legacy(
    conversation: db.Conversation,
    session: Session,
) -> Optional[Dict[str, Any]]:
    roi_detalle = _conversation_roi_details(conversation)
    if roi_detalle:
        return roi_detalle

    form_data = _json_loads(conversation.form_data, {})
    if not isinstance(form_data, dict) or not form_data:
        return None

    normalized_form = normalize_roi_form_fields(form_data)
    probability = estimate_success_probability(normalized_form)
    roi_detalle = build_roi_payload(
        normalized_form,
        probability.get("p_exito"),
        fuente="sistema",
        explicacion=probability.get("explicacion", ""),
    )
    conversation.form_data = json.dumps(normalized_form, ensure_ascii=False)
    conversation.roi_detalle = json.dumps(roi_detalle, ensure_ascii=False)
    conversation.roi = roi_detalle.get("roi")
    session.add(conversation)
    return roi_detalle


def _ensure_potenciadores_for_legacy(
    conversation: db.Conversation,
    session: Session,
) -> Optional[Dict[str, Any]]:
    potenciadores = _conversation_potenciadores(conversation)
    if potenciadores:
        return potenciadores

    form_data = _json_loads(conversation.form_data, {})
    if not isinstance(form_data, dict) or not form_data:
        return None

    potenciadores = build_potenciadores_payload(
        default_business_scores(3),
        estado="sugerido",
        fuente="ia",
        comentario="Evaluación provisional para iniciativa sin score persistido.",
        resumen="Revisar y confirmar criterios en detalle.",
    )
    conversation.potenciadores = json.dumps(potenciadores, ensure_ascii=False)
    session.add(conversation)
    return potenciadores


def _weighted_score(potenciadores: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(potenciadores, dict):
        return None
    value = potenciadores.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_conversation(
    conversation_id: str, session: Session, user_id: int
) -> db.Conversation:
    conversation = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _ensure_conversation_owner(conversation, user_id)
    return conversation


def _get_conversation_for_read(
    conversation_id: str, session: Session, user_id: int, roles: Set[str]
) -> db.Conversation:
    conversation = (
        session.query(db.Conversation)
        .filter(db.Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _ensure_conversation_read_access(conversation, user_id, roles)
    return conversation


def _get_workflow(
    conversation_id: str,
    session: Session,
    user_id: int,
    *,
    create: bool = False,
) -> db.InitiativeWorkflow:
    _get_conversation(conversation_id, session, user_id)
    workflow = (
        session.query(db.InitiativeWorkflow)
        .filter(db.InitiativeWorkflow.conversation_id == conversation_id)
        .first()
    )
    if workflow:
        return workflow
    if not create:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = db.InitiativeWorkflow(
        conversation_id=conversation_id,
        current_status=STATUS_DRAFT,
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
    )
    session.add(workflow)
    session.flush()
    return workflow


def _get_workflow_for_read(
    conversation_id: str, session: Session, user_id: int, roles: Set[str]
) -> db.InitiativeWorkflow:
    _get_conversation_for_read(conversation_id, session, user_id, roles)
    workflow = (
        session.query(db.InitiativeWorkflow)
        .filter(db.InitiativeWorkflow.conversation_id == conversation_id)
        .first()
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


def _get_workflow_for_mutation(
    conversation_id: str,
    session: Session,
    user_id: int,
    roles: Set[str],
    allowed_roles: Set[str],
) -> db.InitiativeWorkflow:
    _get_conversation_for_mutation(
        conversation_id, session, user_id, roles, allowed_roles
    )
    workflow = (
        session.query(db.InitiativeWorkflow)
        .filter(db.InitiativeWorkflow.conversation_id == conversation_id)
        .first()
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


def _assert_status(
    workflow: db.InitiativeWorkflow, allowed_statuses: Set[str]
) -> None:
    current = _normalized_status(workflow.current_status)
    allowed_normalized = {_normalized_status(status) for status in allowed_statuses}
    if current not in allowed_normalized:
        allowed = ", ".join(sorted(allowed_statuses))
        raise HTTPException(
            status_code=409,
            detail=f"Estado inválido: {workflow.current_status}. Esperado: {allowed}.",
        )


def _add_event(
    *,
    session: Session,
    workflow: db.InitiativeWorkflow,
    event_type: str,
    to_status: str,
    actor_user_id: int,
    actor_role: Optional[str],
    actor_name: Optional[str],
    comment: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
) -> db.InitiativeTimelineEvent:
    previous_status = workflow.current_status
    workflow.current_status = to_status
    workflow.updated_by_user_id = actor_user_id
    workflow.updated_at = get_now()

    event = db.InitiativeTimelineEvent(
        workflow_id=workflow.id,
        conversation_id=workflow.conversation_id,
        event_type=event_type,
        from_status=previous_status,
        to_status=to_status,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_name=actor_name,
        comment=comment,
        payload=_json_dumps(payload),
    )
    session.add(event)
    return event


def _calculate_complexity(total_score: int, criteria_count: int) -> Tuple[float, str]:
    average = total_score / criteria_count
    if average <= 2:
        return round(average, 2), "baja"
    if average <= 3.5:
        return round(average, 2), "media"
    return round(average, 2), "alta"


def _serialize_evaluation(
    evaluation: db.InitiativeTechnicalEvaluation,
) -> Dict[str, Any]:
    return {
        "id": evaluation.id,
        "evaluator_user_id": evaluation.evaluator_user_id,
        "evaluator_name": evaluation.evaluator_name,
        "rubric": _json_loads(evaluation.rubric, []),
        "total_score": evaluation.total_score,
        "average_score": evaluation.average_score,
        "complexity": evaluation.complexity,
        "comment": evaluation.comment,
        "created_at": _iso(evaluation.created_at),
    }


def _serialize_event(event: db.InitiativeTimelineEvent) -> Dict[str, Any]:
    return {
        "id": event.id,
        "event_type": event.event_type,
        "from_status": event.from_status,
        "to_status": event.to_status,
        "actor_user_id": event.actor_user_id,
        "actor_role": event.actor_role,
        "actor_name": event.actor_name,
        "comment": event.comment,
        "payload": _json_loads(event.payload, None),
        "created_at": _iso(event.created_at),
    }


def _serialize_workflow(workflow: db.InitiativeWorkflow) -> Dict[str, Any]:
    potenciadores = _conversation_potenciadores(workflow.conversation)
    roi_detalle = _conversation_roi_details(workflow.conversation)
    return {
        "conversation_id": workflow.conversation_id,
        "current_status": _normalized_status(workflow.current_status),
        "created_by_user_id": workflow.created_by_user_id,
        "updated_by_user_id": workflow.updated_by_user_id,
        "created_at": _iso(workflow.created_at),
        "updated_at": _iso(workflow.updated_at),
        "technical_evaluations": [
            _serialize_evaluation(evaluation)
            for evaluation in workflow.technical_evaluations
        ],
        "potenciadores": potenciadores,
        "weighted_score": _weighted_score(potenciadores),
        "roi_detalle": roi_detalle,
        "roi": workflow.conversation.roi if workflow.conversation else None,
    }


def _serialize_workflow_list_item(
    workflow: db.InitiativeWorkflow, conversation: db.Conversation
) -> Dict[str, Any]:
    potenciadores = _conversation_potenciadores(conversation)
    roi_detalle = _conversation_roi_details(conversation)
    return {
        "conversation_id": workflow.conversation_id,
        "initiative_title": conversation.initiative_title,
        "current_status": _normalized_status(workflow.current_status),
        "owner_user_id": conversation.user_id,
        "created_by_user_id": workflow.created_by_user_id,
        "updated_by_user_id": workflow.updated_by_user_id,
        "workflow_created_at": _iso(workflow.created_at),
        "workflow_updated_at": _iso(workflow.updated_at),
        "conversation_created_at": _iso(conversation.created_at),
        "created_at": _iso(workflow.created_at or conversation.created_at),
        "updated_at": _iso(workflow.updated_at or workflow.created_at or conversation.created_at),
        "form_data": normalize_roi_form_fields(_json_loads(conversation.form_data, {})),
        "potenciadores": potenciadores,
        "weighted_score": _weighted_score(potenciadores),
        "roi_detalle": roi_detalle,
        "roi": conversation.roi,
    }


@router.get("", response_model=List[WorkflowListItemResponse])
def list_workflows(
    statuses: Optional[List[str]] = Query(
        default=None,
        description=(
            "Estados a listar. Si se omite, lista iniciativas en committee_ti_review."
        ),
    ),
    owned_only: bool = Query(
        default=True,
        description=(
            "Si es true, lista solo iniciativas del X-User-Id. "
            "Para bandejas globales, Yii debe validar permisos y enviar false."
        ),
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    if not owned_only and not roles.intersection(
        {"ti", "admin"}.union(ROLE_ANY_COMMITTEE_MUTATION)
    ):
        raise HTTPException(
            status_code=403,
            detail="No tienes acceso a la bandeja global de workflows.",
        )

    status_filter = _status_filter_with_legacy(statuses or [STATUS_COMMITTEE_TI])
    query = (
        session.query(db.InitiativeWorkflow, db.Conversation)
        .join(
            db.Conversation,
            db.Conversation.id == db.InitiativeWorkflow.conversation_id,
        )
        .filter(db.InitiativeWorkflow.current_status.in_(status_filter))
    )
    if owned_only:
        query = query.filter(db.Conversation.user_id == user_id)

    rows = (
        query.order_by(desc(db.InitiativeWorkflow.updated_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    needs_commit = False
    items: List[Dict[str, Any]] = []
    for workflow, conversation in rows:
        if not _conversation_potenciadores(conversation):
            if _ensure_potenciadores_for_legacy(conversation, session):
                needs_commit = True
        if not _conversation_roi_details(conversation):
            if _ensure_roi_for_legacy(conversation, session):
                needs_commit = True
        items.append(_serialize_workflow_list_item(workflow, conversation))
    if needs_commit:
        session.commit()
    return items


@router.post("/{conversation_id}/submit-committee", response_model=WorkflowStatusResponse)
def submit_to_committee(
    conversation_id: str,
    data: WorkflowActionInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
):
    workflow = _get_workflow(conversation_id, session, user_id, create=True)
    _assert_status(workflow, {STATUS_DRAFT})
    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_SUBMITTED_TO_COMMITTEE,
        to_status=STATUS_COMMITTEE_TI,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=data.payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/request-technical-review",
    response_model=WorkflowStatusResponse,
)
def request_technical_review(
    conversation_id: str,
    data: WorkflowActionInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id,
        session,
        user_id,
        roles,
        ROLE_ANY_COMMITTEE_MUTATION.union(ROLE_ADMIN_MUTATION),
    )
    _assert_status(workflow, {STATUS_COMMITTEE_TI})
    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_SUBMITTED_TO_OPERATIONS,
        to_status=STATUS_COMMITTEE_OPERATIONS,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=data.payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/technical-evaluations",
    response_model=WorkflowStatusResponse,
)
def submit_technical_evaluation(
    conversation_id: str,
    data: TechnicalEvaluationInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id, session, user_id, roles, ROLE_TI_MUTATION
    )
    _assert_status(workflow, {STATUS_COMMITTEE_TI})

    rubric = [criterion.model_dump() for criterion in data.criteria]
    total_score = sum(criterion.score for criterion in data.criteria)
    average_score, complexity = _calculate_complexity(total_score, len(data.criteria))

    evaluation = db.InitiativeTechnicalEvaluation(
        workflow_id=workflow.id,
        conversation_id=conversation_id,
        evaluator_user_id=user_id,
        evaluator_name=data.evaluator_name,
        rubric=json.dumps(rubric, ensure_ascii=False),
        total_score=total_score,
        average_score=average_score,
        complexity=complexity,
        comment=data.comment,
    )
    session.add(evaluation)
    session.flush()

    event_payload = {
        "technical_evaluation_id": evaluation.id,
        "rubric": rubric,
        "total_score": total_score,
        "average_score": average_score,
        "complexity": complexity,
    }
    if data.payload:
        event_payload["payload"] = data.payload

    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_TECHNICAL_COMMENT_SUBMITTED,
        to_status=STATUS_COMMITTEE_OPERATIONS,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.evaluator_name,
        comment=data.comment,
        payload=event_payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/business-evaluations",
    response_model=WorkflowStatusResponse,
)
def submit_business_evaluation(
    conversation_id: str,
    data: BusinessEvaluationInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id,
        session,
        user_id,
        roles,
        ROLE_ANY_COMMITTEE_MUTATION.union(ROLE_ADMIN_MUTATION),
    )
    _assert_status(
        workflow,
        {
            STATUS_COMMITTEE_TI,
            STATUS_COMMITTEE_OPERATIONS,
            STATUS_COMMITTEE_MANAGEMENT,
        },
    )

    actor_name = data.evaluator_name or f"Usuario {user_id}"
    current_status = _normalized_status(workflow.current_status) or workflow.current_status
    potenciadores = build_potenciadores_payload(
        criteria_to_scores(data.criteria),
        estado="oficial",
        fuente="comite",
        comentario=data.comment,
        explanations=criteria_to_explanations(data.criteria),
        resumen=data.comment or "Score confirmado por comité.",
        actualizado_por=actor_name,
        actualizado_en=_iso(get_now()),
    )

    conversation = workflow.conversation
    conversation.potenciadores = json.dumps(potenciadores, ensure_ascii=False)

    event_payload = {
        "potenciadores": potenciadores,
        "weighted_score": potenciadores.get("score"),
    }
    if data.payload:
        event_payload["payload"] = data.payload

    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_BUSINESS_SCORE_SUBMITTED,
        to_status=current_status,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=actor_name,
        comment=data.comment,
        payload=event_payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/committee-responses",
    response_model=WorkflowStatusResponse,
)
def submit_committee_response(
    conversation_id: str,
    data: CommitteeResponseInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id,
        session,
        user_id,
        roles,
        ROLE_ANY_COMMITTEE_MUTATION.union(ROLE_ADMIN_MUTATION),
    )
    response_type = (data.response_type or "").strip() or "committee_comment"
    current = _normalized_status(workflow.current_status)
    if response_type in {
        "committee_comment",
        "comment",
        "committee_initial_comment",
        "operations_committee_comment",
        "committee_operations_comment",
        "management_committee_comment",
        "committee_management_comment",
        "gerencial_committee_comment",
        "committee_gerencial_comment",
    }:
        _assert_status(
            workflow,
            {
                STATUS_COMMITTEE_TI,
                STATUS_COMMITTEE_OPERATIONS,
                STATUS_COMMITTEE_MANAGEMENT,
            },
        )
        event_type = EVENT_COMMITTEE_COMMENT_SUBMITTED
        to_status = current
    elif response_type == "return_with_observations":
        _assert_status(
            workflow,
            {
                STATUS_COMMITTEE_TI,
                STATUS_COMMITTEE_OPERATIONS,
                STATUS_COMMITTEE_MANAGEMENT,
            },
        )
        event_type = EVENT_RETURNED_WITH_OBSERVATIONS
        if current == STATUS_COMMITTEE_MANAGEMENT:
            to_status = STATUS_COMMITTEE_OPERATIONS
        elif current == STATUS_COMMITTEE_OPERATIONS:
            to_status = STATUS_COMMITTEE_TI
        else:
            to_status = STATUS_DRAFT
    else:
        raise HTTPException(status_code=400, detail="response_type no soportado")

    payload = dict(data.payload or {})
    payload["response_type"] = response_type
    _add_event(
        session=session,
        workflow=workflow,
        event_type=event_type,
        to_status=to_status,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/submit-final-committee",
    response_model=WorkflowStatusResponse,
)
def submit_to_final_committee(
    conversation_id: str,
    data: WorkflowActionInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id,
        session,
        user_id,
        roles,
        ROLE_ANY_COMMITTEE_MUTATION.union(ROLE_ADMIN_MUTATION),
    )
    _assert_status(workflow, {STATUS_COMMITTEE_OPERATIONS})
    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_SUBMITTED_TO_MANAGEMENT,
        to_status=STATUS_COMMITTEE_MANAGEMENT,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=data.payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/reject",
    response_model=WorkflowStatusResponse,
)
def reject_workflow(
    conversation_id: str,
    data: WorkflowActionInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id,
        session,
        user_id,
        roles,
        ROLE_ANY_COMMITTEE_MUTATION.union(ROLE_ADMIN_MUTATION),
    )
    _assert_status(
        workflow,
        {
            STATUS_DRAFT,
            STATUS_COMMITTEE_TI,
            STATUS_COMMITTEE_OPERATIONS,
            STATUS_COMMITTEE_MANAGEMENT,
        },
    )
    if _normalized_status(workflow.current_status) == STATUS_COMMITTEE_MANAGEMENT and "admin" not in roles:
        raise HTTPException(
            status_code=403,
            detail="Solo el rol admin puede revisar/rechazar iniciativas en comité final.",
        )
    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_ANNULLED,
        to_status=STATUS_ANNULLED,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=data.payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.post(
    "/{conversation_id}/approve-final",
    response_model=WorkflowStatusResponse,
)
def approve_final_workflow(
    conversation_id: str,
    data: WorkflowActionInput,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_mutation(
        conversation_id, session, user_id, roles, ROLE_ADMIN_MUTATION
    )
    _assert_status(workflow, {STATUS_COMMITTEE_MANAGEMENT})
    if "admin" not in roles:
        raise HTTPException(
            status_code=403,
            detail="Solo el rol admin puede aprobar iniciativas en comité final.",
        )
    _add_event(
        session=session,
        workflow=workflow,
        event_type=EVENT_SENT_TO_BACKLOG,
        to_status=STATUS_BACKLOG,
        actor_user_id=user_id,
        actor_role=data.actor_role,
        actor_name=data.actor_name,
        comment=data.comment,
        payload=data.payload,
    )
    session.commit()
    session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.get("/{conversation_id}", response_model=WorkflowStatusResponse)
def get_workflow(
    conversation_id: str,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    workflow = _get_workflow_for_read(conversation_id, session, user_id, roles)
    conversation = workflow.conversation
    needs_commit = False
    if conversation and not _conversation_potenciadores(conversation):
        if _ensure_potenciadores_for_legacy(conversation, session):
            needs_commit = True
    if conversation and not _conversation_roi_details(conversation):
        if _ensure_roi_for_legacy(conversation, session):
            needs_commit = True
    if needs_commit:
        session.commit()
        session.refresh(workflow)
    return _serialize_workflow(workflow)


@router.get(
    "/{conversation_id}/timeline",
    response_model=List[TimelineEventResponse],
)
def get_timeline(
    conversation_id: str,
    session: Session = Depends(db.get_db),
    user_id: int = Depends(_require_user_id),
    roles: Set[str] = Depends(_get_rbac_roles),
):
    _get_workflow_for_read(conversation_id, session, user_id, roles)
    events = (
        session.query(db.InitiativeTimelineEvent)
        .filter(db.InitiativeTimelineEvent.conversation_id == conversation_id)
        .order_by(db.InitiativeTimelineEvent.created_at.asc())
        .all()
    )
    return [_serialize_event(event) for event in events]
