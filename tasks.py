import os
from celery import Celery
from celery.result import AsyncResult

# Importar os serviços que a tarefa irá usar
from services.database import DatabaseManager
from services.embedding import CLIPEmbedder
from services.image_processor import ImageProcessor
from core.config import settings

# --- Configuração do Celery ---
# O Celery usará as variáveis de ambiente que definimos no docker-compose.yml
celery_app = Celery(
    "tasks",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# --- Instâncias de Serviço para o Worker ---
# O worker precisa de suas próprias instâncias dos serviços para se conectar ao DB e carregar o modelo.
# Como o worker é um processo separado, ele não compartilha memória com a API.
db_manager = DatabaseManager()
embedder = CLIPEmbedder()
image_processor = ImageProcessor(db_manager, embedder)

@celery_app.on_after_configure.connect
def setup_future_tasks(sender, **kwargs):
    """Garante que o worker possa se conectar ao banco de dados."""
    # O loop de eventos do Celery é diferente, então precisamos de uma maneira
    # de executar a corrotina de conexão do asyncpg.
    import asyncio
    asyncio.run(db_manager.connect_pool())

# --- Definição da Tarefa ---
@celery_app.task(bind=True)
def process_folder_task(self, folder_path: str, use_augmentation: bool):
    """
    A tarefa Celery que executa o processamento de imagens de forma assíncrona.
    O 'self' (bind=True) nos dá acesso ao objeto da tarefa para atualizar o estado.
    """
    try:
        # Usar a lógica que já tínhamos no ImageProcessor, mas adaptada para Celery
        from pathlib import Path
        import time

        folder = Path(folder_path)
        all_image_files = [p for p in folder.rglob('*') if p.suffix.lower() in settings.IMAGE_EXTS]
        total_images = len(all_image_files)

        if total_images == 0:
            raise ValueError("No images found in the specified folder.")

        print(f"Task {self.request.id}: Found {total_images} images.")
        self.update_state(state='PROGRESS', meta={'progress': f"0/{total_images}"})

        # O worker precisa de seu próprio loop de eventos para executar as corrotinas
        loop = asyncio.get_event_loop()
        
        # A lógica de processamento de imagem agora é chamada aqui
        loop.run_until_complete(
            image_processor.process_images_in_folder_celery(self, total_images, all_image_files, use_augmentation)
        )

        return {'progress': f"{total_images}/{total_images}", 'result': f"Successfully processed {total_images} images."}

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        print(f"Task {self.request.id}: Failed. Reason: {str(e)}")
        # Re-raise a exceção para que o Celery a capture como uma falha
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
        info['error'] = str(task_result.info) # A exceção é armazenada em info
    
    return info
