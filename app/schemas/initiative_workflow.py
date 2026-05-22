from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class WorkflowActionInput(BaseModel):
    comment: Optional[str] = None
    actor_role: Optional[str] = None
    actor_name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class CommitteeResponseInput(WorkflowActionInput):
    response_type: str = "user_review"


class TechnicalCriterionInput(BaseModel):
    criterion: str
    score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class TechnicalEvaluationInput(BaseModel):
    criteria: List[TechnicalCriterionInput] = Field(..., min_length=1)
    comment: Optional[str] = None
    evaluator_name: Optional[str] = None
    actor_role: Optional[str] = "ti"
    payload: Optional[Dict[str, Any]] = None


BusinessCriterionName = Literal[
    "financiero",
    "estrategico",
    "cliente",
    "datos_ia",
    "time_to_value",
    "complejidad",
    "riesgo",
]


class BusinessCriterionInput(BaseModel):
    criterion: BusinessCriterionName
    score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class BusinessEvaluationInput(BaseModel):
    criteria: List[BusinessCriterionInput] = Field(..., min_length=7)
    comment: Optional[str] = None
    evaluator_name: Optional[str] = None
    actor_role: Optional[str] = "iniciativa_comite"
    payload: Optional[Dict[str, Any]] = None


class TechnicalEvaluationResponse(BaseModel):
    id: int
    evaluator_user_id: int
    evaluator_name: Optional[str]
    rubric: List[TechnicalCriterionInput]
    total_score: int
    average_score: float
    complexity: str
    comment: Optional[str]
    created_at: Optional[str]


class TimelineEventResponse(BaseModel):
    id: int
    event_type: str
    from_status: Optional[str]
    to_status: str
    actor_user_id: int
    actor_role: Optional[str]
    actor_name: Optional[str]
    comment: Optional[str]
    payload: Optional[Dict[str, Any]]
    created_at: Optional[str]


class WorkflowStatusResponse(BaseModel):
    conversation_id: str
    current_status: str
    created_by_user_id: int
    updated_by_user_id: Optional[int]
    created_at: Optional[str]
    updated_at: Optional[str]
    technical_evaluations: List[TechnicalEvaluationResponse]
    potenciadores: Optional[Dict[str, Any]] = None
    weighted_score: Optional[float] = None
    roi_detalle: Optional[Dict[str, Any]] = None
    roi: Optional[float] = None


class WorkflowListItemResponse(BaseModel):
    conversation_id: str
    initiative_title: Optional[str]
    current_status: str
    owner_user_id: Optional[int]
    created_by_user_id: int
    updated_by_user_id: Optional[int]
    workflow_created_at: Optional[str]
    workflow_updated_at: Optional[str]
    conversation_created_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    form_data: Dict[str, Any]
    potenciadores: Optional[Dict[str, Any]] = None
    weighted_score: Optional[float] = None
    roi_detalle: Optional[Dict[str, Any]] = None
    roi: Optional[float] = None
