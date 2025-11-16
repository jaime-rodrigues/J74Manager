import os
import io
import numpy as np
from fastapi import APIRouter, Form, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Literal

# Import schemas and the new Celery task
from api.schemas import EmbeddingRequest, SearchResponse, ListImagesResponse, TaskCreationResponse
from tasks import process_folder_task

router = APIRouter(
    prefix="/images",
    tags=["Images"]
)

@router.post("/process-folder/", response_model=TaskCreationResponse)
async def process_folder_endpoint(
    request: Request,
    folder: str = Form(...),
    use_augmentation: bool = Form(False, description="Se True, gera embeddings para versões aumentadas de cada imagem.")
):
    """
    Starts a persistent, trackable background task via Celery to process a folder of images.
    Returns a task ID to monitor the progress.
    """
    from core.config import settings
    target_folder = os.path.join(settings.UPLOAD_DIR, folder)
    
    if not os.path.isdir(target_folder):
        raise JSONResponse(status_code=404, content={"message": f"Folder '{folder}' not found in upload directory."})

    # Dispara a tarefa Celery. O Celery irá serializar os argumentos e enviá-los para o Redis.
    task = process_folder_task.delay(target_folder, use_augmentation)
    
    status_endpoint = request.url_for('get_task_status', task_id=task.id)
    # Converte o objeto URL para string para compatibilidade com Pydantic V2
    return TaskCreationResponse(task_id=task.id, status_endpoint=str(status_endpoint))

# ... (os outros endpoints de busca e listagem não precisam de grandes mudanças) ...
@router.post("/search-by-embedding", response_model=SearchResponse)
async def search_by_embedding_endpoint(request: Request, embedding_req: EmbeddingRequest, scope: Literal['original_only', 'all'] = Query('original_only')):
    from services.database import DatabaseManager
    db_manager = DatabaseManager() # Instância temporária para a busca
    await db_manager.connect_pool()
    try:
        embedding = np.array(embedding_req.embedding, dtype=np.float32)
        results = await db_manager.search_similar(embedding, top_k=embedding_req.top_k, scope=scope)
        if not results: return JSONResponse(status_code=404, content={"message": "No similar images found."})
        return SearchResponse(similar_images=results, query_embedding=embedding_req.embedding)
    finally:
        await db_manager.close_pool()

@router.post("/search-by-upload", response_model=SearchResponse)
async def search_by_upload_endpoint(request: Request, file: UploadFile = File(...), top_k: int = Form(5, gt=0, le=100), scope: Literal['original_only', 'all'] = Form('original_only')):
    from services.database import DatabaseManager
    from services.embedding import CLIPEmbedder
    db_manager = DatabaseManager()
    embedder = CLIPEmbedder()
    await db_manager.connect_pool()
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        embedding = embedder.generate_embedding(image)
        results = await db_manager.search_similar(embedding, top_k=top_k, scope=scope)
        if not results: return JSONResponse(status_code=404, content={"message": "No similar images found."})
        return SearchResponse(query_filename=file.filename, similar_images=results, query_embedding=embedding.tolist())
    finally:
        await db_manager.close_pool()

@router.get("/", response_model=ListImagesResponse)
async def list_images_endpoint(request: Request, limit: int = 100, offset: int = 0):
    from services.database import DatabaseManager
    db_manager = DatabaseManager()
    await db_manager.connect_pool()
    try:
        records = await db_manager.list_records(limit=limit, offset=offset)
        return ListImagesResponse(images=records)
    finally:
        await db_manager.close_pool()
