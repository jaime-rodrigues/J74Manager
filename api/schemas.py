from pydantic import BaseModel, Field
from typing import List

# --- Request Schemas ---

class EmbeddingRequest(BaseModel):
    """Schema for a search request using a pre-computed embedding."""
    embedding: List[float]
    top_k: int = Field(5, gt=0, le=100, description="Number of similar images to return.")

class ImageSearchRequest(BaseModel):
    """Schema for a search request using an image file (will be deprecated)."""
    top_k: int = Field(5, gt=0, le=100)


# --- Response Schemas ---

class ImageResult(BaseModel):
    """Schema for a single image result in a similarity search."""
    id: int
    filename: str
    filepath: str
    similarity: float

    class Config:
        orm_mode = True

class ImageRecord(BaseModel):
    """Schema for a single image record from the database listing."""
    id: int
    filename: str
    filepath: str

    class Config:
        orm_mode = True

class SearchResponse(BaseModel):
    """Schema for the response of a similarity search."""
    query_filename: str | None = None
    similar_images: List[ImageResult]

class ListImagesResponse(BaseModel):
    """Schema for the response of listing images."""
    images: List[ImageRecord]
