from pydantic import BaseModel, Field
from typing import List, Any

# --- Task Schemas ---

class Task(BaseModel):
    """Schema for representing the status of a background task."""
    id: str
    status: str
    progress: str | None = None
    result: Any | None = None
    error: str | None = None

class TaskCreationResponse(BaseModel):
    """Schema for the response when a new task is created."""
    task_id: str
    status_endpoint: str

# --- Image Schemas (existentes) ---

class EmbeddingRequest(BaseModel):
    embedding: List[float]
    top_k: int = Field(5, gt=0, le=100)

class ImageResult(BaseModel):
    id: int
    filename: str
    filepath: str
    similarity: float
    class Config:
        from_attributes = True

class ImageRecord(BaseModel):
    id: int
    filename: str
    filepath: str
    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    query_filename: str | None = None
    similar_images: List[ImageResult]
    query_embedding: List[float] | None = None

class ListImagesResponse(BaseModel):
    images: List[ImageRecord]
