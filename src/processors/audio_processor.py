from typing import List, Any
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, AutoModel
import tempfile

from .base_processor import BaseProcessor
from ..utils.types import ContentItem, ModalityType, GranularityLevel

class AudioProcessor(BaseProcessor):
    """Processor for audio content using audio-specific transformers"""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(ModalityType.AUDIO)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def _segment_audio(self, waveform: torch.Tensor, sample_rate: int, granularity: GranularityLevel) -> List[torch.Tensor]:
        """Segment audio based on granularity"""
        # Calculate segment sizes based on granularity
        if granularity == GranularityLevel.FINE:
            # 3-second segments
            segment_size = 3 * sample_rate
        elif granularity == GranularityLevel.MEDIUM:
            # 10-second segments
            segment_size = 10 * sample_rate
        else:  # COARSE
            # 30-second segments
            segment_size = 30 * sample_rate
            
        # Split audio into segments
        segments = []
        for i in range(0, waveform.size(1), segment_size):
            segment = waveform[:, i:i + segment_size]
            if segment.size(1) >= sample_rate:  # Only keep segments of at least 1 second
                segments.append(segment)
                
        return segments
        
    def process(self, raw_content: str, granularity: GranularityLevel) -> List[ContentItem]:
        """Process audio file into segments based on granularity level"""
        if not os.path.exists(raw_content):
            raise ValueError(f"Audio file not found: {raw_content}")
            
        # Load audio
        waveform, sample_rate = torchaudio.load(raw_content)
        
        # Segment audio
        segments = self._segment_audio(waveform, sample_rate, granularity)
        
        # Create content items for each segment
        content_items = []
        for i, segment in enumerate(segments):
            # Save segment temporarily
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                torchaudio.save(temp_file.name, segment, sample_rate)
                
                content_items.append(ContentItem(
                    id=f"{os.path.basename(raw_content)}_{i}",
                    modality=self.modality,
                    granularity=granularity,
                    content=temp_file.name,  # Store path to audio segment
                    metadata={
                        "source_audio": raw_content,
                        "segment_number": i,
                        "total_segments": len(segments),
                        "duration": segment.size(1) / sample_rate
                    },
                    embedding=None
                ))
                
        return content_items
        
    def embed(self, content_item: ContentItem) -> List[float]:
        """Generate embeddings for an audio segment"""
        if content_item.modality != self.modality:
            raise ValueError(f"Cannot process content of type {content_item.modality}")
            
        # Load the audio segment
        waveform, sample_rate = torchaudio.load(content_item.content)
        
        # Process the audio
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over time dimension
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
        return embeddings[0].tolist()
