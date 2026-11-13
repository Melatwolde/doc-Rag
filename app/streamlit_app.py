"""streamlit_app.py

Streamlit Demonstration Interface for DocuRAG Assistant.
Integrates hybrid retrieval, reranking/generation, and abstention guardrails.
"""

from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

# Add src to Python path so package imports resolve correctly.
root_dir = Path(__file__).resolve().parent.parent
src_path = root_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from docurag.pipeline import HybridRAGPipeline

st.set_page_config(
    page_title="DocuRAG | Intelligent Documentation Assistant",
    page_icon=None,
    layout="wide",
)

# Custom CSS for a professional, clean AI interface styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.03em;
        font-size: 2.25rem !important;
        margin-bottom: 0.2rem;
    }
    .stTextInput input {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.015);
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.15);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header (Names removed)
st.title("DocuRAG Assistant")
st.markdown(
    "**Hybrid Neural & Lexical Retrieval System** combining BM25 lexical search, dense embeddings, "
    "Reciprocal Rank Fusion, cross-encoder reranking, and confidence guardrails."
)
st.markdown("---")

# Sidebar Controls for Pipeline Configuration
st.sidebar.header("Pipeline Configuration")
st.sidebar.markdown("Tune retrieval weights and confidence thresholds.")

confidence_threshold = st.sidebar.slider(
    "Abstention Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="Minimum confidence score required to prevent system abstention."
)
top_k_retrieve = st.sidebar.slider(
    "Retrieval Top-K", min_value=3, max_value=20, value=10, step=1,
    help="Number of candidate chunks retrieved via hybrid RRF search."
)
top_k_rerank = st.sidebar.slider(
    "Reranked Top-K", min_value=1, max_value=5, value=3, step=1,
    help="Number of top passages passed to the generator."
)
use_cross_encoder = st.sidebar.checkbox(
    "Use Cross-Encoder Reranker",
    value=True,
    help="The first search may download the cross-encoder model if it is not cached.",
)

# Verify processed chunks file exists before loading pipeline
chunks_file = root_dir / "data" / "processed" / "chunks.json"
if not chunks_file.exists():
    st.error(f"Processed chunks not found at `{chunks_file}`. Please run the ingestion pipeline first.")
    st.stop()

@st.cache_resource
def load_rag_pipeline(
    threshold: float,
    k_ret: int,
    k_rerank: int,
    use_cross_encoder: bool,
) -> HybridRAGPipeline:
    reranker = None
    if use_cross_encoder:
        from docurag.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    return HybridRAGPipeline(
        chunks_path=chunks_file,
        confidence_threshold=threshold,
        top_k_retrieve=k_ret,
        top_k_rerank=k_rerank,
        cross_encoder_reranker=reranker,
        use_cross_encoder=use_cross_encoder,
    )

try:
    pipeline = load_rag_pipeline(
        confidence_threshold,
        top_k_retrieve,
        top_k_rerank,
        use_cross_encoder,
    )
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

# Main Search Input
query = st.text_input(
    "Query Documentation",
    placeholder="e.g., How can I implement JWT authentication and refresh an expired token?",
    label_visibility="collapsed",
)

col_input_spacer, col_search_btn = st.columns([3, 1])
with col_search_btn:
    search_triggered = st.button("Generate Answer", type="primary", use_container_width=True)

if search_triggered:
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Executing Hybrid RAG Pipeline..."):
            try:
                response = pipeline.run(query.strip())
            except Exception as e:
                st.error(f"Failed to execute RAG pipeline: {e}")
                st.stop()

        st.markdown("---")

        if response["is_abstaining"]:
            st.error("System Abstention Triggered (ERR_INSUFFICIENT_EVIDENCE)")
            st.info(response["abstention_reason"])
            st.warning("The documentation does not contain sufficient verified evidence to answer this query safely.")
        else:
            st.success("Grounded Answer Generated Successfully")

            col1, col2 = st.columns([3, 2], gap="large")

            with col1:
                st.subheader("Generated Answer")
                st.caption("Extractive answer assembled from cited documentation passages.")
                st.markdown(response["answer"])
                st.caption(f"End-to-end latency: {response.get('latency_ms', 0.0):.2f} ms")

                st.markdown("### Citations & Sources")
                for cit in response.get("citations", []):
                    section = cit.get("heading_path") or cit.get("section_title") or "Documentation section"
                    location = ""
                    if cit.get("line_start") is not None:
                        location = f" (lines {cit['line_start']}–{cit.get('line_end', cit['line_start'])})"
                    if cit.get("source_url"):
                        st.markdown(f"- [{section}]({cit['source_url']}){location} — `{cit['chunk_id']}`")
                    else:
                        st.markdown(f"- {section}{location} — `{cit['chunk_id']}` (local source)")

            with col2:
                st.subheader(
                    "Retrieved & Cross-Encoder Reranked Passages"
                    if use_cross_encoder
                    else "Retrieved Evidence Passages"
                )
                reranked = response.get("reranked_chunks", [])
                for idx, chunk in enumerate(reranked, 1):
                    metadata = chunk.get("metadata", {})
                    section_title = chunk.get("section_title") or metadata.get("section_title") or "Section"
                    score = float(
                        chunk.get("score", 0.0)
                        if use_cross_encoder
                        else chunk.get("confidence_score", 0.0)
                    )
                    score_label = "Cross-encoder score" if use_cross_encoder else "Confidence"
                    with st.expander(f"[{idx}] {section_title} ({score_label}: {score:.4f})"):
                        st.code(chunk.get("content", ""), language="markdown")
                        st.caption(f"Chunk ID: `{chunk.get('chunk_id')}`")
                        if metadata.get("source_url"):
                            st.link_button("Open source section", metadata["source_url"])
