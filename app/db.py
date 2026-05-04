import os
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mariadb+pymysql://ai_user:ai_password@host.docker.internal:3306/talento_humano",
)

ENGINE_OPTIONS = {"pool_pre_ping": True}
if DATABASE_URL.startswith(("mysql", "mariadb")):
    ENGINE_OPTIONS["connect_args"] = {"connect_timeout": 5}

engine = create_engine(DATABASE_URL, **ENGINE_OPTIONS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversaciones_iniciativas"

    id = Column(String(36), primary_key=True)
    initiative_title = Column(String(255), index=True)
    form_data = Column(LONGTEXT, nullable=True)  # Stores JSON of the full form state
    # Propietario en la app Yii (Yii user id). NULL = filas creadas antes de esta columna o sin rellenar.
    user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "mensajes_iniciativas"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversaciones_iniciativas.id"))
    role = Column(String(50))  # "user" or "agent"
    content = Column(LONGTEXT)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
