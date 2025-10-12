import os
import io
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Import schemas and services
from api.schemas import EmbeddingRequest, SearchResponse, ListImagesResponse
from core.config import settings
from services.database import DatabaseManager
from services.embedding import CLIPEmbedder
from services.image_processor import ImageProcessor

router = APIRouter(
    prefix="/images",
    tags=["Images"]
)

# Placeholder for dependency injection
def get_services():
    from main import db_manager, embedder, image_processor
    return db_manager, embedder, image_processor

@router.post("/process-folder/")
async def process_folder_endpoint(background_tasks: BackgroundTasks, folder: str = Form(...)):
    """
    Asynchronously processes a folder of images in the background.
    The folder path must be relative to the configured UPLOAD_DIR.
    """
    _, _, image_processor = get_services()
    target_folder = os.path.join(settings.UPLOAD_DIR, folder)
    
    if not os.path.isdir(target_folder):
        return JSONResponse(status_code=404, content={"message": f"Folder '{folder}' not found in upload directory."})

    background_tasks.add_task(image_processor.process_images_in_folder, target_folder)
    return {"message": f"Processing started for folder: {folder}. This may take some time."}

@router.post("/search-by-embedding", response_model=SearchResponse)
async def search_by_embedding_endpoint(request: EmbeddingRequest):
    """
    Performs similarity search using a pre-computed embedding vector.
    This is the most direct and efficient way to search.
    """
    db_manager, _, _ = get_services()
    try:
        embedding = np.array(request.embedding, dtype=np.float32)
        
        # The database call is async and returns the data in the correct structure
        results = await db_manager.search_similar(embedding, top_k=request.top_k)
        
        if not results:
            return JSONResponse(status_code=404, content={"message": "No similar images found."})
            
        return SearchResponse(similar_images=results)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@router.post("/search-by-upload", response_model=SearchResponse)
async def search_by_upload_endpoint(file: UploadFile = File(...), top_k: int = Form(5, gt=0, le=100)):
    """
    Uploads an image, generates its embedding, and then performs a similarity search.
    This is a convenience endpoint for clients that don't generate embeddings themselves.
    """
    db_manager, embedder, _ = get_services()
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        embedding = embedder.generate_embedding(image)
        results = await db_manager.search_similar(embedding, top_k=top_k)
        
        if not results:
            return JSONResponse(status_code=404, content={"message": "No similar images found."})
            
        return SearchResponse(query_filename=file.filename, similar_images=results)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@router.get("/", response_model=ListImagesResponse)
async def list_images_endpoint(limit: int = 100, offset: int = 0):
    """
    Asynchronously lists the images currently indexed in the database,
    with pagination support.
    """
    db_manager, _, _ = get_services()
    records = await db_manager.list_records(limit=limit, offset=offset)
    return ListImagesResponse(images=records)
