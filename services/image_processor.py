import os
import asyncio
from pathlib import Path
from typing import List
from PIL import Image
import torchvision.transforms as T

from core.config import settings
from services.database import DatabaseManager
from services.embedding import CLIPEmbedder

class ImageProcessor:
    """Handles asynchronous image processing, augmentation, and folder scanning."""
    def __init__(self, db_manager: DatabaseManager, embedder: CLIPEmbedder):
        self.db_manager = db_manager
        self.embedder = embedder
        self.transformations = [
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]

    def _apply_transformations(self, image: Image.Image) -> List[Image.Image]:
        """Applies a series of augmentations to an image."""
        transformed_images = [transform(image) for transform in self.transformations]
        transformed_images.append(image)  # Always include the original image
        return transformed_images

    async def process_images_in_folder(self, folder_path: str):
        """
        Asynchronously scans a folder for images, generates embeddings, and saves
        them to the database in batches.
        """
        print(f"Starting to process folder: {folder_path}")
        folder = Path(folder_path)
        all_image_files = [p for p in folder.rglob('*') if p.suffix.lower() in settings.IMAGE_EXTS]
        total_images = len(all_image_files)
        print(f"Found {total_images} images to process.")

        records = []
        for i, image_file in enumerate(all_image_files):
            try:
                # Image loading and embedding generation are CPU/IO-bound but not async native.
                # For very high throughput, you might run them in a separate process pool.
                # For this use case, we keep it simple.
                image = Image.open(image_file)
                augmented_images = self._apply_transformations(image)

                for aug_image in augmented_images:
                    embedding = self.embedder.generate_embedding(aug_image)
                    relative_path = os.path.relpath(image_file, settings.UPLOAD_DIR)
                    # The database manager now expects the numpy array directly
                    records.append((os.path.basename(image_file), relative_path, embedding))

                # Asynchronously insert a batch when it's full
                if len(records) >= settings.BATCH_SIZE:
                    print(f"Inserting batch of {len(records)} embeddings...")
                    await self.db_manager.insert_embeddings_batch(records)
                    records = []
                
                print(f"Processed {i + 1}/{total_images}: {image_file}")

            except Exception as e:
                print(f"Skipping {image_file} due to error: {e}")

        # Asynchronously insert any remaining records
        if records:
            print(f"Inserting final batch of {len(records)} embeddings...")
            await self.db_manager.insert_embeddings_batch(records)
        
        print("Folder processing complete.")
