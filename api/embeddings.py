from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import List

# Import services
from services.database import DatabaseManager

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"]
)

# Dependency placeholder to get service instances from main.py
def get_db_manager() -> DatabaseManager:
    from main import db_manager
    return db_manager

@router.post("/get-by-filepaths")
async def get_embeddings_by_filepaths(filepaths: List[str] = Form(...)):
    """
    Receives a list of filepaths and returns a dictionary mapping each filepath
    to its corresponding embedding vector.
    """
    db_manager = get_db_manager()
    try:
        embeddings_map = {}
        for path in filepaths:
            embedding = await db_manager.get_embedding_by_filepath(path)
            if embedding is not None:
                # Convert numpy array to a standard list for JSON serialization
                embeddings_map[path] = embedding.tolist()
            else:
                embeddings_map[path] = None
        
        return embeddings_map

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
