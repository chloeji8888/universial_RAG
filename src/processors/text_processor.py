from typing import List, Optional
import uuid
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set this before importing any HuggingFace modules

from nltk.tokenize import sent_tokenize, word_tokenize

from .base_processor import BaseProcessor
from ..utils.types import ContentItem, ModalityType, GranularityLevel

class TextProcessor(BaseProcessor):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__(ModalityType.TEXT)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def process(self, raw_content: str, granularity: GranularityLevel) -> List[ContentItem]:
        if granularity == GranularityLevel.FINE:
            # Split into sentences
            chunks = sent_tokenize(raw_content)
        elif granularity == GranularityLevel.MEDIUM:
            # Split into paragraphs
            chunks = [p.strip() for p in raw_content.split('\n\n') if p.strip()]
        else:  # COARSE
            # Keep as single document
            chunks = [raw_content]
            
        return [
            ContentItem(
                id=str(uuid.uuid4()),
                modality=self.modality,
                granularity=granularity,
                content=chunk,
                embedding=None
            )
            for chunk in chunks
        ]
    
    def embed(self, content_item: ContentItem) -> List[float]:
        # Tokenize and encode text
        inputs = self.tokenizer(
            content_item.content,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling to get sentence embedding
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
        return embeddings[0].tolist()
