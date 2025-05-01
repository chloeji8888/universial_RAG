from typing import List, Optional, Tuple, Dict
import numpy as np
from collections import defaultdict

from .base_retriever import BaseRetriever
from ..utils.types import ContentItem, Query, ModalityType, GranularityLevel

class VectorRetriever(BaseRetriever):
    def __init__(self, modality: ModalityType):
        super().__init__(modality)
        # Store items by granularity for efficient retrieval
        self.items_by_granularity: Dict[GranularityLevel, List[ContentItem]] = defaultdict(list)
        self.embeddings_by_granularity: Dict[GranularityLevel, np.ndarray] = {}
    
    def add_items(self, items: List[ContentItem]) -> None:
        for item in items:
            if item.embedding is None:
                raise ValueError("Items must have embeddings before being added to the retriever")
            self.items_by_granularity[item.granularity].append(item)
        
        # Update embedding matrices
        for granularity in self.items_by_granularity:
            embeddings = [item.embedding for item in self.items_by_granularity[granularity]]
            self.embeddings_by_granularity[granularity] = np.array(embeddings)
    
    def retrieve(self, query: Query, top_k: int = 5) -> List[Tuple[ContentItem, float]]:
        if query.embedding is None:
            raise ValueError("Query must have an embedding")
            
        granularity = query.granularity or GranularityLevel.MEDIUM
        if granularity not in self.embeddings_by_granularity:
            return []
            
        # Compute similarities
        query_embedding = np.array(query.embedding)
        similarities = np.dot(self.embeddings_by_granularity[granularity], query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return items and scores
        results = []
        for idx in top_indices:
            item = self.items_by_granularity[granularity][idx]
            score = float(similarities[idx])
            results.append((item, score))
            
        return results
