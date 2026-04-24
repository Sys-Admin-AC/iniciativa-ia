from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Comprobación para load balancers / Kubernetes (sin depender de BD)."""
    return {"status": "ok", "service": "iniciativa-ai"}
