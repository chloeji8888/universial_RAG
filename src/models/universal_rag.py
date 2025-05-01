from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..utils.types import (
    ContentItem, Query, ModalityType, GranularityLevel, 
    RoutingDecision, Corpus
)
from ..processors.base_processor import BaseProcessor
from ..retrievers.base_retriever import BaseRetriever

class UniversalRAG:
    def __init__(
        self,
        processors: Dict[ModalityType, BaseProcessor],
        retrievers: Dict[ModalityType, BaseRetriever],
        llm_model: Any  # This could be a specific LLM interface
    ):
        self.processors = processors
        self.retrievers = retrievers
        self.llm_model = llm_model
        
        # Store corpora by modality and granularity
        self.corpora: Dict[ModalityType, Dict[GranularityLevel, Corpus]] = defaultdict(dict)
    
    def add_content(self, raw_content: Any, modality: ModalityType, granularity: GranularityLevel) -> None:
        """Add new content to the system"""
        if modality not in self.processors:
            raise ValueError(f"No processor available for modality: {modality}")
            
        # Process content into items
        processor = self.processors[modality]
        items = processor.process(raw_content, granularity)
        
        # Generate embeddings
        for item in items:
            item.embedding = processor.embed(item)
        
        # Add to corpus
        if granularity not in self.corpora[modality]:
            self.corpora[modality][granularity] = Corpus(modality, granularity, [])
        self.corpora[modality][granularity].items.extend(items)
        
        # Add to retriever
        self.retrievers[modality].add_items(items)
    
    def _route_query(self, query: Query) -> RoutingDecision:
        """Determine which corpus to use for retrieval.

        The logic follows three stages (all *train-free*):

        1.  **Explicit override** – if the caller already provided a
            modality / granularity in the ``Query`` object we obey it.
        2.  **Keyword matching** – richer keyword dictionaries for each
            routing target.  This is still heuristic but far more
            comprehensive than the original few words.
        3.  **Fallback** – if none of the above fires we default to
            paragraph-level text retrieval (the safest, most granular).
        """

        text_lower = query.text.lower()

        # ------------------------------------------------------------------
        # 1. caller-specified override
        # ------------------------------------------------------------------
        if query.modality is not None:
            if query.modality == ModalityType.TEXT:
                if query.granularity == GranularityLevel.COARSE:
                    return RoutingDecision.DOCUMENT
                return RoutingDecision.PARAGRAPH  # fine or medium – use MEDIUM
            if query.modality == ModalityType.IMAGE:
                return RoutingDecision.IMAGE
            if query.modality == ModalityType.VIDEO:
                if query.granularity == GranularityLevel.FINE:
                    return RoutingDecision.CLIP
                return RoutingDecision.VIDEO

        # ------------------------------------------------------------------
        # 2. keyword-based automatic routing
        # ------------------------------------------------------------------
        # If the query is not asking a factual question, skip retrieval
        non_retrieval_phrases = [
            "hello", "hi", "good morning", "how are you", "thanks", "thank you"
        ]
        if any(p in text_lower for p in non_retrieval_phrases):
            return RoutingDecision.NONE

        # Define richer keyword dictionaries
        image_kw = [
            "image", "picture", "photo", "diagram", "figure", "chart", "graph",
            "visual", "screenshot", "map", "illustration"
        ]
        clip_kw = [
            "clip", "segment", "shot", "timestamp", "part", "scene"
        ]
        video_kw = [
            "video", "footage", "film", "animation", "watch", "playback"
        ]
        doc_coarse_kw = [
            "overall", "summary", "whole", "entire", "document", "article",
            "paper", "overview", "high level"
        ]
        paragraph_kw = [
            "paragraph", "section", "sentence", "detail", "specific",
            "exact", "quote"
        ]

        # Perform hierarchical matching
        if any(k in text_lower for k in clip_kw):
            return RoutingDecision.CLIP
        if any(k in text_lower for k in video_kw):
            return RoutingDecision.VIDEO
        if any(k in text_lower for k in image_kw):
            return RoutingDecision.IMAGE
        if any(k in text_lower for k in doc_coarse_kw):
            return RoutingDecision.DOCUMENT
        if any(k in text_lower for k in paragraph_kw):
            return RoutingDecision.PARAGRAPH

        # ------------------------------------------------------------------
        # 3. sensible default
        # ------------------------------------------------------------------
        return RoutingDecision.PARAGRAPH
    
    def _get_corpus(self, decision: RoutingDecision) -> Optional[Corpus]:
        """Get the appropriate corpus based on routing decision"""
        if decision == RoutingDecision.NONE:
            return None
            
        mapping = {
            RoutingDecision.PARAGRAPH: (ModalityType.TEXT, GranularityLevel.MEDIUM),
            RoutingDecision.DOCUMENT: (ModalityType.TEXT, GranularityLevel.COARSE),
            RoutingDecision.IMAGE: (ModalityType.IMAGE, GranularityLevel.MEDIUM),
            RoutingDecision.CLIP: (ModalityType.VIDEO, GranularityLevel.FINE),
            RoutingDecision.VIDEO: (ModalityType.VIDEO, GranularityLevel.COARSE),
        }
        
        if decision not in mapping:
            return None
            
        modality, granularity = mapping[decision]
        return self.corpora.get(modality, {}).get(granularity)
    
    def process_query(
        self,
        query_text: str,
        modality: Optional[ModalityType] = None,
        granularity: Optional[GranularityLevel] = None,
        top_k: int = 5
    ) -> str:
        """Process a query and generate a response"""
        # Create query object
        query = Query(
            text=query_text,
            modality=modality,
            granularity=granularity
        )
        
        # Route query to appropriate corpus
        decision = self._route_query(query)
        corpus = self._get_corpus(decision)
        
        # Debug: show routing information
        if decision == RoutingDecision.NONE or corpus is None:
            print(f"[Router] Decision: {decision.name} -> No retrieval; answering directly with LLM.")
        else:
            print(
                f"[Router] Decision: {decision.name} -> Modality: {corpus.modality.value}, "
                f"Granularity: {corpus.granularity.name}, Items in corpus: {len(corpus.items)}"
            )
        
        if decision == RoutingDecision.NONE or corpus is None:
            # Generate response without retrieval
            return self._generate_response(query_text, "No retrieval", [])
        
        # Process query with appropriate processor
        processor = self.processors[corpus.modality]
        query_items = processor.process(query_text, corpus.granularity)
        query_item = query_items[0]  # Take first chunk for simplicity
        query_item.embedding = processor.embed(query_item)
        
        # Retrieve relevant content
        retrieved = self.retrievers[corpus.modality].retrieve(query_item, top_k=top_k)
        
        # Format context for the LLM (text + vision blocks)
        context_str, context_blocks = self._format_context(retrieved)

        # Generate response using LLM
        response = self._generate_response(query_text, context_str, context_blocks)
        
        return response
    
    def _format_context(self, retrieved_content: List[Tuple[ContentItem, float]]) -> Tuple[str, List[dict]]:
        """Return (plain_text_context, multimodal_blocks[]) for the LLM.

        multimodal_blocks follows the OpenAI vision spec – a list where each
        element is either a text block: {"type": "text", "text": ...} or an
        image block: {"type": "image_url", "image_url": {"url": data_uri}}.
        """

        import base64, mimetypes, os

        text_parts: List[str] = []
        blocks: List[dict] = []

        for item, score in retrieved_content:
            if item.modality == ModalityType.TEXT:
                segment = f"[Text] {item.content}"
                text_parts.append(segment)
                blocks.append({"type": "text", "text": segment})
            elif item.modality == ModalityType.IMAGE:
                # convert local image to data URI
                path = item.content
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "image/jpeg"
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                data_uri = f"data:{mime};base64,{b64}"
                blocks.append({"type": "text", "text": "[Image]"})
                blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
                text_parts.append(f"[Image] {item.metadata}")
            else:
                # fallback for unsupported types
                text_parts.append(f"[{item.modality.value}] {item.metadata}")
                blocks.append({"type": "text", "text": f"[{item.modality.value}] {item.metadata}"})

        return "\n\n".join(text_parts), blocks
    
    def _generate_response(self, query: str, context_str: str, context_blocks: List[dict]) -> str:
        """Generate response using the LLM – text-only or multimodal."""

        # If we have any image blocks -> build multimodal message payload
        if any(block.get("type") == "image_url" for block in context_blocks):
            # Combine context blocks with the query as final text block for user
            user_content = context_blocks + [{"type": "text", "text": f"Query: {query}"}]
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant that answers questions based on provided context (text and images)."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            return self.llm_model.generate(messages)
        else:
            prompt = f"""Based on the following context, please answer the query.

Context:
{context_str}

Query: {query}

Response:"""
            return self.llm_model.generate(prompt)
