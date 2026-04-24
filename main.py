"""Punto de entrada compatible con `uvicorn main:app` (reexporta la app del paquete `app`)."""

from app.main import app

__all__ = ["app"]
