import os
import sys
import tempfile
from typing import List

import streamlit as st
import graphviz                                  # NEW

# Ensure project root on PYTHONPATH so we can import src.* modules when app is
# executed from anywhere (``streamlit run`` does an internal cd).
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.test import build_system, load_corpus  # reuse helper functions
from src.utils.types import ModalityType, GranularityLevel
from src.retrievers.vector_retriever import VectorRetriever  # NEW

st.set_page_config(page_title="Universal RAG Demo", page_icon="🤖", layout="centered")

st.title("🤖 Universal RAG – Multimodal Retrieval-Augmented Generation")

# -----------------------------------------------------------------------------
# Create / retrieve the UniversalRAG instance and seed corpus once per session
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_rag():
    rag = build_system()
    load_corpus(rag)
    return rag

rag = get_rag()

# -----------------------------------------------------------------------------
# Sidebar – allow user to ingest additional content on the fly
# -----------------------------------------------------------------------------
st.sidebar.header("Add content to corpus")

modality_label = st.sidebar.selectbox("Modality", ["Text", "Image"], index=0)
granularity_label = st.sidebar.selectbox(
    "Granularity", [g.name.title() for g in GranularityLevel], index=1
)

granularity = GranularityLevel[granularity_label.upper()]

if modality_label == "Text":
    text_content = st.sidebar.text_area("Paste text here…", height=150)
    if st.sidebar.button("➕ Add Text") and text_content.strip():
        rag.add_content(text_content, ModalityType.TEXT, granularity)
        st.sidebar.success("Text added to corpus ✅")
else:
    uploaded_file = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if st.sidebar.button("➕ Add Image") and uploaded_file is not None:
        # -------------------------------------------------------------
        # Clear any previously stored IMAGE content so that retrieval
        # only considers the *latest* upload (avoiding stale history).
        # -------------------------------------------------------------
        rag.corpora[ModalityType.IMAGE] = {}
        rag.retrievers[ModalityType.IMAGE] = VectorRetriever(modality=ModalityType.IMAGE)

        # Save to a temporary path – RAG expects a file path for images
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(uploaded_file.read())
        tmp.close()

        rag.add_content(tmp.name, ModalityType.IMAGE, granularity)
        st.sidebar.success("Image added to corpus ✅ (cleared previous images)")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ How to use this demo"):
    st.write(
        "1. Ask any question in the main input box.\n"
        "2. The router will decide which modality / granularity to search.\n"
        "3. The retrieved context (text + images) is fed to GPT-4o-mini.\n"
        "4. Optionally add more text or images from the sidebar before asking."
    )

# -----------------------------------------------------------------------------
# Main – chat-like interface for querying the RAG system
# -----------------------------------------------------------------------------
query = st.text_input("Ask me anything…", placeholder="e.g. What are the key features of Universal RAG?", key="user_query")

if st.button("🔍 Run Query", type="primary") and query.strip():
    with st.spinner("Thinking…"):
        answer, dbg = rag.process_query(query, return_debug=True)

    # ------------------- 1️⃣ show answer -------------------
    st.markdown("### 💬 Answer")
    st.write(answer)

    # ------------------- 2️⃣ show workflow -----------------
    with st.expander("🗺️  Workflow (click to expand)", expanded=False):
        # a) flowchart
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR", fontsize="10")
        dot.node("Q", "User Query")
        dot.node("R", f"Router\n({dbg['decision'].name})")
        dot.node("V", f"Retriever\n({dbg['modality'].value.capitalize()})")
        dot.node("L", "LLM")
        dot.node("A", "Answer")
        dot.edges(["QR", "RV", "VL", "LA"])
        st.graphviz_chart(dot, use_container_width=True)

        # b) metadata table
        st.markdown("**Routing details**")
        st.json(
            {
                "RoutingDecision": dbg["decision"].name,
                "Modality": dbg["modality"].value if dbg["modality"] else None,
                "Granularity": dbg["granularity"].name if dbg["granularity"] else None,
                "Top-k": len(dbg["retrieved"]),
            }
        )

        # c) retrieved context
        st.markdown("**Retrieved Context**")
        for idx, (item, score) in enumerate(dbg["retrieved"], 1):
            st.caption(f"{idx}. score={score:.3f}")
            if item.modality == ModalityType.TEXT:
                st.write(item.content)
            elif item.modality == ModalityType.IMAGE:
                st.image(item.content, caption=item.metadata or f"image {idx}")
            else:
                st.write(f"[{item.modality.value}] {item.metadata}")
    st.markdown("---")
    st.caption("Powered by Universal RAG + GPT-4o-mini") 