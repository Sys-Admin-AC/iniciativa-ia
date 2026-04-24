import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://ai_user:ai_password@postgres:5432/th_ai")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)  # Using UUID string for simplicity from frontend
    initiative_title = Column(String, index=True)
    form_data = Column(Text, nullable=True)  # Stores JSON of the full form state
    # Propietario en la app Yii (Yii user id). NULL = filas creadas antes de esta columna o sin rellenar.
    user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "agent"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


def _ensure_conversations_user_id_column() -> None:
    """Añade user_id a tablas ya existentes (create_all no altera columnas)."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id INTEGER"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations (user_id)"
                )
            )
    except Exception as e:
        logging.warning("No se pudo asegurar columna user_id: %s", e)


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_conversations_user_id_column()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
