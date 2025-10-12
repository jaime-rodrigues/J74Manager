import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from retry import retry

from core.config import settings

class CLIPEmbedder:
    """Handles the generation of image embeddings using a CLIP model."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing CLIPEmbedder on device: {self.device}")
        self.model = CLIPModel.from_pretrained(settings.CLIP_BASE).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_BASE)

    @retry(tries=3, delay=1)
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generates a vector embedding for a single image."""
        try:
            # Ensure image is in RGB format, as required by CLIP
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            return features.cpu().numpy().squeeze().astype(np.float32)
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
