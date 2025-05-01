from typing import List, Any
import os
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import tempfile

from .base_processor import BaseProcessor
from ..utils.types import ContentItem, ModalityType, GranularityLevel

class VideoProcessor(BaseProcessor):
    """Processor for video content using video-specific transformers"""
    
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32"):
        super().__init__(ModalityType.VIDEO)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def _extract_frames(self, video_path: str, granularity: GranularityLevel) -> List[np.ndarray]:
        """Extract frames from video based on granularity"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame sampling rate based on granularity
        if granularity == GranularityLevel.FINE:
            # Extract frames every second
            sample_rate = int(fps)
        elif granularity == GranularityLevel.MEDIUM:
            # Extract frames every 5 seconds
            sample_rate = int(fps * 5)
        else:  # COARSE
            # Extract frames every 15 seconds
            sample_rate = int(fps * 15)
            
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_count += 1
            
        cap.release()
        return frames
        
    def process(self, raw_content: str, granularity: GranularityLevel) -> List[ContentItem]:
        """Process video file into frames based on granularity level"""
        if not os.path.exists(raw_content):
            raise ValueError(f"Video file not found: {raw_content}")
            
        # Extract frames
        frames = self._extract_frames(raw_content, granularity)
        
        # Create content items for each frame/segment
        content_items = []
        for i, frame in enumerate(frames):
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                frame_pil = Image.fromarray(frame)
                frame_pil.save(temp_file.name)
                
                content_items.append(ContentItem(
                    id=f"{os.path.basename(raw_content)}_{i}",
                    modality=self.modality,
                    granularity=granularity,
                    content=temp_file.name,  # Store path to frame
                    metadata={
                        "source_video": raw_content,
                        "frame_number": i,
                        "total_frames": len(frames)
                    },
                    embedding=None
                ))
                
        return content_items
        
    def embed(self, content_item: ContentItem) -> List[float]:
        """Generate embeddings for a video frame"""
        if content_item.modality != self.modality:
            raise ValueError(f"Cannot process content of type {content_item.modality}")
            
        # Load the frame
        image = Image.open(content_item.content)
        
        # Process the frame
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embeddings = outputs.pooler_output
            
        return embeddings[0].tolist()
