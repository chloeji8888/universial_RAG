from typing import List, Any
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import uuid

from .base_processor import BaseProcessor
from ..utils.types import ContentItem, ModalityType, GranularityLevel

class ImageProcessor(BaseProcessor):
    """Processor for image content using vision transformers (e.g., CLIP)"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(ModalityType.IMAGE)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def process(self, raw_content: Any, granularity: GranularityLevel) -> List[ContentItem]:
        """Convert raw input (image *or* text query) into a `ContentItem`.

        If `raw_content` is an existing file / PIL.Image -> treat as image.
        Otherwise assume it is a textual query in natural language that we
        want to embed into the same joint space (CLIP text encoder).
        """

        # Case 1 ────────── raw_content is an image path or PIL.Image
        if isinstance(raw_content, str) and os.path.exists(raw_content):
            image_path = raw_content
            return [
                ContentItem(
                    id=str(uuid.uuid4()),
                    modality=self.modality,
                    granularity=granularity,
                    content=image_path,
                    metadata={"path": image_path, "is_text_query": False},
                    embedding=None
                )
            ]

        if isinstance(raw_content, Image.Image):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                raw_content.save(tmp.name)
                image_path = tmp.name
            return [
                ContentItem(
                    id=str(uuid.uuid4()),
                    modality=self.modality,
                    granularity=granularity,
                    content=image_path,
                    metadata={"path": image_path, "is_text_query": False},
                    embedding=None
                )
            ]

        # Case 2 ────────── treat as natural language query describing image
        if isinstance(raw_content, str):
            return [
                ContentItem(
                    id=str(uuid.uuid4()),
                    modality=self.modality,  # Still IMAGE modality for compatibility
                    granularity=granularity,
                    content=raw_content,
                    metadata={"is_text_query": True},
                    embedding=None
                )
            ]

        raise ValueError("raw_content must be a valid image path, PIL.Image, or text string")

    def embed(self, content_item: ContentItem) -> List[float]:
        if content_item.modality != self.modality:
            raise ValueError("Cannot embed non-image content with ImageProcessor")

        # Decide whether this is a text query or an actual image
        is_text = content_item.metadata and content_item.metadata.get("is_text_query", False)

        with torch.no_grad():
            if is_text:
                # Use CLIP text encoder
                inputs = self.processor(text=[content_item.content], return_tensors="pt", padding=True, truncation=True)
                embeddings = self.model.get_text_features(**inputs)
            else:
                image = Image.open(content_item.content).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                embeddings = self.model.get_image_features(**inputs)

        return embeddings[0].tolist() 