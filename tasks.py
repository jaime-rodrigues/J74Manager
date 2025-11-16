import os
import asyncio
from celery import Celery
from celery.result import AsyncResult

# Importar os serviços que a tarefa irá usar
from services.database import DatabaseManager
from services.embedding import CLIPEmbedder
from services.image_processor import ImageProcessor
from core.config import settings

# --- Configuração do Celery ---
celery_app = Celery(
    "tasks",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# --- Instâncias de Serviço para o Worker ---
# O embedder pode ser global, pois é carregado uma vez e depois é somente leitura.
# Isso economiza memória e tempo de carregamento em cada tarefa.
embedder = CLIPEmbedder()

# O DatabaseManager e o ImageProcessor serão instanciados DENTRO da tarefa
# para garantir que cada tarefa tenha seu próprio pool de conexões de banco de dados.

# --- Definição da Tarefa ---
@celery_app.task(bind=True)
def process_folder_task(self, folder_path: str, use_augmentation: bool):
    """
    Tarefa Celery que processa imagens. Cada execução desta tarefa gerencia
    seu próprio ciclo de vida de conexão com o banco de dados para evitar conflitos de concorrência.
    """
    # 1. Instanciar serviços por tarefa para garantir isolamento.
    db_manager = DatabaseManager()
    image_processor = ImageProcessor(db_manager, embedder)

    async def _process():
        # 2. Conectar ao banco de dados no início da execução da tarefa.
        await db_manager.connect_pool()
        try:
            from pathlib import Path
            folder = Path(folder_path)
            all_image_files = [p for p in folder.rglob('*') if p.suffix.lower() in settings.IMAGE_EXTS]
            total_images = len(all_image_files)

            if total_images == 0:
                raise ValueError("No images found in the specified folder.")

            print(f"Task {self.request.id}: Found {total_images} images.")
            self.update_state(state='PROGRESS', meta={'progress': f"0/{total_images}"})

            # A lógica de processamento principal
            await image_processor.process_images_in_folder_celery(
                self, total_images, all_image_files, use_augmentation
            )

            return {'progress': f"{total_images}/{total_images}", 'result': f"Successfully processed {total_images} images."}
        finally:
            # 3. Garantir que a conexão seja fechada, mesmo se ocorrer um erro.
            print(f"Task {self.request.id}: Closing database connections.")
            await db_manager.disconnect_pool()

    try:
        # 4. Executar a lógica assíncrona principal.
        result = asyncio.run(_process())
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        print(f"Task {self.request.id}: Failed. Reason: {str(e)}")
        raise

def get_task_info(task_id: str) -> dict:
    """Consulta o backend do Celery (Redis) para obter o status da tarefa."""
    task_result = AsyncResult(task_id, app=celery_app)
    
    info = {
        "id": task_id,
        "status": task_result.state,
        "progress": None,
        "result": None,
        "error": None
    }

    if task_result.state == 'PENDING':
        info['progress'] = "Aguardando na fila..."
    elif task_result.state == 'PROGRESS':
        info.update(task_result.info)
    elif task_result.state == 'SUCCESS':
        info.update(task_result.result)
    elif task_result.state == 'FAILURE':
        info['error'] = str(task_result.info)
    
    return info
