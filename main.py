import os
from fastapi import FastAPI
from pathlib import Path

# Import configurations and routers
from core.config import settings
from api import database as db_router
from api import images as img_router
from api import metrics as metrics_router
from api import embeddings as embeddings_router
from api import diagnostics as diagnostics_router
from api import tasks as tasks_router

# --- FastAPI Application Setup ---
# Não precisamos mais instanciar os serviços aqui, pois a API e o Worker
# criarão suas próprias instâncias conforme necessário.
app = FastAPI(
    title="Image Embedding and Search API (Professional)",
    description="A professional API using Celery and Redis for persistent, trackable background tasks.",
    version="3.0.0"
)

# Os eventos de startup/shutdown da API não precisam mais gerenciar o DB pool,
# pois cada requisição (ou o worker) gerenciará seu próprio ciclo de vida de conexão.
# Isso torna a API mais stateless e robusta.

# --- API Routers ---
app.include_router(db_router.router)
app.include_router(img_router.router)
app.include_router(metrics_router.router)
app.include_router(embeddings_router.router)
app.include_router(diagnostics_router.router)
app.include_router(tasks_router.router)

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"message": "Image Embedding API is running."}

# Ensure __init__.py files exist
Path("api/__init__.py").touch(exist_ok=True)
Path("core/__init__.py").touch(exist_ok=True)
Path("services/__init__.py").touch(exist_ok=True)
