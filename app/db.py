import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.utils import get_now

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mariadb+pymysql://root:root@192.168.1.112:3306/talento_humano",
)

ENGINE_OPTIONS = {"pool_pre_ping": True}
if DATABASE_URL.startswith(("mysql", "mariadb")):
    ENGINE_OPTIONS["connect_args"] = {"connect_timeout": 5}

engine = create_engine(DATABASE_URL, **ENGINE_OPTIONS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "iniciativa_conversaciones"

    id = Column(String(36), primary_key=True)
    initiative_title = Column(String(255), index=True)
    form_data = Column(LONGTEXT, nullable=True)  # Stores JSON of the full form state
    potenciadores = Column(LONGTEXT, nullable=True)
    # Propietario en la app Yii (Yii user id). NULL = filas creadas antes de esta columna o sin rellenar.
    user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=get_now)

    messages = relationship("Message", back_populates="conversation")
    workflow = relationship(
        "InitiativeWorkflow", back_populates="conversation", uselist=False
    )


class Message(Base):
    __tablename__ = "iniciativa_mensajes"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36), ForeignKey("iniciativa_conversaciones.id"))
    role = Column(String(50))  # "user" or "agent"
    content = Column(LONGTEXT)
    created_at = Column(DateTime, default=get_now)

    conversation = relationship("Conversation", back_populates="messages")


class InitiativeWorkflow(Base):
    __tablename__ = "iniciativa_workflows"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        String(36),
        ForeignKey("iniciativa_conversaciones.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    current_status = Column(String(80), nullable=False, index=True)
    created_by_user_id = Column(Integer, nullable=False, index=True)
    updated_by_user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=get_now)
    updated_at = Column(DateTime, default=get_now, onupdate=get_now)

    conversation = relationship("Conversation", back_populates="workflow")
    events = relationship(
        "InitiativeTimelineEvent",
        back_populates="workflow",
        order_by="InitiativeTimelineEvent.created_at",
    )
    technical_evaluations = relationship(
        "InitiativeTechnicalEvaluation",
        back_populates="workflow",
        order_by="InitiativeTechnicalEvaluation.created_at",
    )


class InitiativeTimelineEvent(Base):
    __tablename__ = "iniciativa_timeline_events"

    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey("iniciativa_workflows.id"), nullable=False)
    conversation_id = Column(String(36), nullable=False, index=True)
    event_type = Column(String(80), nullable=False, index=True)
    from_status = Column(String(80), nullable=True)
    to_status = Column(String(80), nullable=False, index=True)
    actor_user_id = Column(Integer, nullable=False, index=True)
    actor_role = Column(String(80), nullable=True, index=True)
    actor_name = Column(String(255), nullable=True)
    comment = Column(LONGTEXT, nullable=True)
    payload = Column(LONGTEXT, nullable=True)
    created_at = Column(DateTime, default=get_now, index=True)

    workflow = relationship("InitiativeWorkflow", back_populates="events")


class InitiativeTechnicalEvaluation(Base):
    __tablename__ = "iniciativa_evaluaciones_ti"

    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey("iniciativa_workflows.id"), nullable=False)
    conversation_id = Column(String(36), nullable=False, index=True)
    evaluator_user_id = Column(Integer, nullable=False, index=True)
    evaluator_name = Column(String(255), nullable=True)
    rubric = Column(LONGTEXT, nullable=False)
    total_score = Column(Integer, nullable=False)
    average_score = Column(Float, nullable=False)
    complexity = Column(String(40), nullable=False, index=True)
    comment = Column(LONGTEXT, nullable=True)
    created_at = Column(DateTime, default=get_now, index=True)

    workflow = relationship(
        "InitiativeWorkflow", back_populates="technical_evaluations"
    )


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
