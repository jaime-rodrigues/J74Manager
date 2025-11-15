import io
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Import services
from services.metrics import MetricCalculator
from services.embedding import CLIPEmbedder
from services.database import DatabaseManager

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"]
)

# Dependency placeholder to get service instances from main.py
def get_services():
    from main import db_manager, embedder
    return db_manager, embedder

# Instantiate the calculator
metric_calculator = MetricCalculator()

@router.post("/compare-embeddings")
async def compare_embeddings_endpoint(
    query_image: UploadFile = File(...),
    selected_filepath: str = Form(...)
):
    """
    Calculates semantic similarity metrics between two images by comparing their embeddings.
    - Generates embedding for the uploaded query image.
    - Fetches the pre-calculated embedding for the selected image from the database.
    """
    db_manager, embedder = get_services()
    try:
        # 1. Generate embedding for the uploaded query image
        query_contents = await query_image.read()
        pil_query_image = Image.open(io.BytesIO(query_contents))
        query_embedding = embedder.generate_embedding(pil_query_image)

        # 2. Fetch the embedding for the selected image from the database
        selected_embedding = await db_manager.get_embedding_by_filepath(selected_filepath)

        if selected_embedding is None:
            return JSONResponse(
                status_code=404, 
                content={"message": f"Embedding not found in database for filepath: {selected_filepath}"}
            )

        # 3. Calculate all metrics between the two embeddings
        report = metric_calculator.calculate_all_metrics(query_embedding, selected_embedding)

        return report

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred during metric calculation: {str(e)}"})
