from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..utils.types import ContentItem, ModalityType, GranularityLevel

class BaseProcessor(ABC):
    """Base class for all modality-specific processors"""
    
    def __init__(self, modality: ModalityType):
        self.modality = modality
    
    @abstractmethod
    def process(self, raw_content: Any, granularity: GranularityLevel) -> List[ContentItem]:
        """Process raw content into a list of ContentItems at specified granularity.
        
        Args:
            raw_content (Any): Raw content to be processed. Type depends on modality:
                - TEXT: str
                - IMAGE: PIL.Image or path to image
                - VIDEO: path to video file
                - AUDIO: path to audio file
            granularity (GranularityLevel): Level of content segmentation:
                - FINE: smallest meaningful units (sentences, image patches)
                - MEDIUM: medium-sized units (paragraphs, full images)
                - COARSE: largest units (documents, video clips)
                
        Returns:
            List[ContentItem]: List of processed content items with unique IDs and metadata.
                Each item will have embedding=None until embed() is called.
        """
        pass
    
    @abstractmethod
    def embed(self, content_item: ContentItem) -> List[float]:
        """Generate embeddings for a content item.
        
        Args:
            content_item (ContentItem): Content item to generate embeddings for.
                The content_item.modality must match self.modality.
                
        Returns:
            List[float]: Dense vector representation of the content item.
                The dimensionality depends on the underlying embedding model.
                
        Raises:
            ValueError: If content_item.modality does not match self.modality
        """
        if content_item.modality != self.modality:
            raise ValueError(f"Processor {self.modality} cannot process content of type {content_item.modality}")
        return content_item.embedding
