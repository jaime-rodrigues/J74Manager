import os
from pathlib import Path
from typing import List
from PIL import Image
import torchvision.transforms as T
import time

from core.config import settings
from services.database import DatabaseManager
from services.embedding import CLIPEmbedder

class ImageProcessor:
    """Handles asynchronous image processing, designed to be called by a Celery task."""
    def __init__(self, db_manager: DatabaseManager, embedder: CLIPEmbedder):
        self.db_manager = db_manager
        self.embedder = embedder
        self.transformations = [
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]

    def _apply_transformations(self, image: Image.Image) -> List[Image.Image]:
        """Applies a series of augmentations to an image."""
        return [transform(image) for transform in self.transformations]

    async def process_images_in_folder_celery(
        self, 
        task, # O objeto da tarefa Celery para atualizar o estado
        total_images: int,
        all_image_files: List[Path],
        use_augmentation: bool = False
    ):
        """
        The core logic for processing images, now designed to be called by a Celery worker.
        """
        records = []
        for i, image_file in enumerate(all_image_files):
            try:
                image = Image.open(image_file).convert("RGB")
                relative_path = os.path.relpath(image_file, settings.UPLOAD_DIR)
                
                original_embedding = self.embedder.generate_embedding(image)
                records.append((os.path.basename(image_file), relative_path, original_embedding, 'original'))

                if use_augmentation:
                    augmented_images = self._apply_transformations(image)
                    for aug_image in augmented_images:
                        aug_embedding = self.embedder.generate_embedding(aug_image)
                        records.append((os.path.basename(image_file), relative_path, aug_embedding, 'augmented'))

                if len(records) >= settings.BATCH_SIZE:
                    await self.db_manager.insert_embeddings_batch(records)
                    records = []
                
                # Update task progress
                progress_text = f"{i + 1}/{total_images}"
                task.update_state(state='PROGRESS', meta={'progress': progress_text})
                print(f"Task {task.request.id}: Progress {progress_text}")
                time.sleep(0.01)

            except Exception as e:
                # Log o erro mas continua o processo se possível
                print(f"Task {task.request.id}: Skipping {image_file} due to error: {e}")

        if records:
            await self.db_manager.insert_embeddings_batch(records)
