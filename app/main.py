import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db, qdrant_client_setup
from app.api.routers import conversations, health, history, workflow

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Initiative Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    db.init_db()
    qdrant_client_setup.init_qdrant_collection()


app.include_router(health.router)
app.include_router(conversations.router)
app.include_router(history.router)
app.include_router(workflow.router)
