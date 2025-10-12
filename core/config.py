import os
from typing import Tuple

class Settings:
    """Holds all application settings."""
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://colab:colab@db:5432/colab")
    IMAGE_EXTS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')
    BATCH_SIZE: int = 32
    EMBEDDING_DIM: int = 768
    CLIP_BASE: str = "openai/clip-vit-large-patch14"
    UPLOAD_DIR: str = "/app/uploads"
    BACKUP_DIR: str = "/app/backups"

settings = Settings()
