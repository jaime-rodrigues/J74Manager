import io
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

# Import services
from services.embedding import CLIPEmbedder
from services.database import DatabaseManager
from services.metrics import MetricCalculator

router = APIRouter(
    prefix="/diagnostics",
    tags=["Diagnostics"]
)

# Dependency placeholder
def get_services():
    from main import db_manager, embedder
    return db_manager, embedder

metric_calculator = MetricCalculator()

@router.post("/compare-embeddings")
async def diagnostic_compare_embeddings(
    upload_image: UploadFile = File(...),
    db_filepath: str = Form(...)
):
    """
    Provides a detailed comparison between an uploaded image's embedding and a
    stored embedding from the database for diagnostic purposes.
    """
    db_manager, embedder = get_services()
    try:
        # 1. Generate a fresh embedding for the uploaded image
        contents = await upload_image.read()
        pil_image = Image.open(io.BytesIO(contents))
        new_embedding = embedder.generate_embedding(pil_image)

        # 2. Fetch the stored embedding from the database
        # Note: This might fetch one of several augmented embeddings associated with the filepath
        stored_embedding = await db_manager.get_embedding_by_filepath(db_filepath)

        if stored_embedding is None:
            return JSONResponse(
                status_code=404, 
                content={"message": f"No embedding found in database for filepath: {db_filepath}"}
            )

        # 3. Calculate semantic metrics between the two embeddings
        semantic_report = metric_calculator.calculate_all_metrics(new_embedding, stored_embedding)

        # 4. Prepare a detailed report
        report = {
            "info": "Comparação entre o embedding gerado no upload e um embedding armazenado no DB.",
            "uploaded_image_embedding_preview": new_embedding.tolist()[:10], # Preview first 10 dims
            "stored_image_embedding_preview": stored_embedding.tolist()[:10], # Preview first 10 dims
            "semantic_comparison_report": semantic_report
        }

        return report

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred during diagnostics: {str(e)}"})
