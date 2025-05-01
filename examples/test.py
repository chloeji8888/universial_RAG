import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"       # HF noise suppression
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from openai import OpenAI                             
from src.utils.types import ModalityType, GranularityLevel
from src.processors.text_processor import TextProcessor
from src.processors.image_processor import ImageProcessor
from src.retrievers.vector_retriever import VectorRetriever
from src.models.universal_rag import UniversalRAG

# ---------- tiny OpenAI helper ----------
class OpenAILLM:
    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()                    
        self.client = OpenAI()
        self.model = model
    def generate(self, prompt_or_messages):
        """Generate text using the underlying OpenAI chat model.

        This helper accepts either a **plain prompt string** or a **fully
        constructed list of chat messages** (useful for multimodal / vision |
        advanced scenarios).  If the caller supplies a list we forward it
        verbatim to the OpenAI client; otherwise we wrap the string inside a
        single user message.
        """

        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
        else:
            messages = [{"role": "user", "content": str(prompt_or_messages)}]

        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        return rsp.choices[0].message.content.strip()

def build_system() -> UniversalRAG:
    text_processor  = TextProcessor()
    image_processor = ImageProcessor()
    text_retriever  = VectorRetriever(modality=ModalityType.TEXT)
    image_retriever = VectorRetriever(modality=ModalityType.IMAGE)
    llm             = OpenAILLM()
    return UniversalRAG(
        processors={
            ModalityType.TEXT: text_processor,
            ModalityType.IMAGE: image_processor
        },
        retrievers={
            ModalityType.TEXT: text_retriever,
            ModalityType.IMAGE: image_retriever
        },
        llm_model=llm
    )

def load_corpus(rag: UniversalRAG):
    docs = [
        """Universal RAG is a framework for retrieval-augmented generation that
        supports multiple modalities and multiple granularity levels.  It can
        retrieve paragraphs, documents, video clips, etc.""",

        """Its router decides, for each user query, which corpus (paragraph, doc,
        image, clip, …) should be consulted before answering.  In our demo we
        mimic the router with simple rules over the query text."""
    ]
    # add paragraph-level chunks
    for d in docs:
        rag.add_content(d, ModalityType.TEXT, GranularityLevel.MEDIUM)
    # add a coarse, full-document representation
    rag.add_content("\n\n".join(docs), ModalityType.TEXT, GranularityLevel.COARSE)

    # add a sample image (adjust path)
    # img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")   # make sure this file exists
    # or give an absolute path:
    img_path = "/Users/chloe/Desktop/universial_RAG/examples/sample.jpg"

    if os.path.exists(img_path):
        rag.add_content(img_path, ModalityType.IMAGE, GranularityLevel.MEDIUM)
    else:
        raise FileNotFoundError(img_path)

def main():
    rag = build_system()
    load_corpus(rag)

    queries = [
        "What are the key features of Universal RAG?",
        "Give me an overall summary of Universal RAG.",
        "Hello!",
        "Show me the picture we stored, and tell me what is the color of the picture."
    ]
    for q in queries:
        print(f"\nUSER: {q}")
        print("RAG :", rag.process_query(q))

if __name__ == "__main__":
    main()