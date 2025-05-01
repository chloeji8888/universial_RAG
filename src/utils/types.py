from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

class ModalityType(Enum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    CLIP = "clip"
    VIDEO = "video"
    NONE = "none"
    # AUDIO = "audio"
    

class GranularityLevel(Enum):
    FINE = "fine"      # e.g., sentences, image patches, video clips
    MEDIUM = "medium"  # e.g., paragraphs, full images
    COARSE = "coarse" # e.g., documents, full videos

class RoutingDecision(Enum):
    NONE = "None"
    PARAGRAPH = "Paragraph"
    DOCUMENT = "Document"
    IMAGE = "Image"
    CLIP = "Clip"
    VIDEO = "Video"
    # AUDIO = "Audio"

@dataclass
class ContentItem:
    """Base class for all content items regardless of modality"""
    id: str
    modality: ModalityType
    granularity: GranularityLevel
    content: Any
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

@dataclass
class Query:
    """Represents a query with its modality and other attributes"""
    text: str
    modality: Optional[ModalityType] = None
    granularity: Optional[GranularityLevel] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

@dataclass
class Corpus:
    """Represents a corpus of content items with specific modality and granularity"""
    modality: ModalityType
    granularity: GranularityLevel
    items: List[ContentItem]
