import os
from fastapi import FastAPI
from pathlib import Path

# Import configurations, routers, and service classes
from core.config import settings
from api import database as db_router
from api import images as img_router
from services.database import DatabaseManager
from services.backup import BackupManager
from services.embedding import CLIPEmbedder
from services.image_processor import ImageProcessor

# --- Service Instantiation ---
db_manager = DatabaseManager()
backup_manager = BackupManager(db_manager)
embedder = CLIPEmbedder()
image_processor = ImageProcessor(db_manager, embedder)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Image Embedding and Search API (Async)",
    description="An asynchronous API for processing images, generating vector embeddings, and performing similarity searches using asyncpg and pgvector.",
    version="2.1.0" # Version bump for the fix
)

@app.on_event("startup")
async def startup_event():
    """Asynchronous startup event to initialize services."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.BACKUP_DIR, exist_ok=True)

    print("Initializing database...")
    latest_backup = backup_manager.get_latest_backup()
    
    if latest_backup:
        print(f"Found latest backup: {latest_backup}. Attempting to restore...")
        try:
            # This process now handles closing/re-opening the pool and ensuring the extension exists.
            await backup_manager.restore_database(latest_backup)
            print(f"Database successfully restored from {latest_backup}.")
        except Exception as e:
            print(f"CRITICAL: Error restoring database: {e}. Initializing a new database as a fallback.")
            # If restore fails, just create a fresh pool and schema.
            await db_manager.connect_pool()
            await db_manager.create_table()
    else:
        print("No backup found. Initializing a new database.")
        # connect_pool now ensures the vector extension exists before creating the pool.
        await db_manager.connect_pool()
        await db_manager.create_table()

@app.on_event("shutdown")
async def shutdown_event():
    """Asynchronous shutdown event to close the database connection pool."""
    await db_manager.close_pool()

# --- API Routers ---
app.include_router(db_router.router)
app.include_router(img_router.router)

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"message": "Image Embedding API is running."}

# Ensure __init__.py files exist
Path("api/__init__.py").touch(exist_ok=True)
Path("core/__init__.py").touch(exist_ok=True)
Path("services/__init__.py").touch(exist_ok=True)
