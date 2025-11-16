import os
import asyncio
from fastapi import FastAPI
from pathlib import Path

# Import configurations and routers
from core.config import settings
from services.database import DatabaseManager
from api import database as db_router
from api import images as img_router
from api import metrics as metrics_router
from api import embeddings as embeddings_router
from api import diagnostics as diagnostics_router
from api import tasks as tasks_router

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Image Embedding and Search API (Professional)",
    description="A professional API using Celery and Redis for persistent, trackable background tasks.",
    version="3.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    On startup, connect to the database and create the necessary tables.
    This ensures the 'image_embeddings' table exists before any operations are performed.
    """
    print("API is starting up...")
    db_manager = DatabaseManager()
    await db_manager.connect_pool()
    await db_manager.create_table()
    # We don't close the pool here, as it might be used by other parts of the app
    # during its lifecycle. FastAPI will handle the shutdown.
    print("API startup complete. Database table is ready.")

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
