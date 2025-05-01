from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..utils.types import ContentItem, Query, ModalityType, GranularityLevel

class BaseRetriever(ABC):
    """Base class for retrievers"""
    
    def __init__(self, modality: ModalityType):
        self.modality = modality
    
    @abstractmethod
    def add_items(self, items: List[ContentItem]) -> None:
        """Add content items to the retriever's index"""
        pass
    
    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 5) -> List[Tuple[ContentItem, float]]:
        """Retrieve most relevant items for the query with their scores"""
        pass
