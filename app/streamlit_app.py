"""
CDSS Streamlit Application

PURPOSE:
Main entry point for the Clinical Decision Support System web interface.
Provides a chat-based interface for clinical queries with source citations.

USAGE:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.rag.pipeline import CDSSPipeline
from src.database.db_operations import get_document_count, get_unique_sources
from src.api.ollama_client import check_ollama_status, list_models
from src.config.settings import settings

from app.components.chat_interface import (
    initialize_chat_history,
    add_message,
    clear_chat_history,
    render_chat_history,
    render_chat_input
)
from app.components.source_display import render_knowledge_base_stats, render_sources


# === Page Configuration ===
st.set_page_config(
    page_title="CDSS - Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .status-ok {
        color: #4CAF50;
    }
    .status-error {
        color: #F44336;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# === Initialize Session State ===
initialize_chat_history()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# === Sidebar ===
with st.sidebar:
    st.markdown("## 🏥 CDSS")
    st.markdown("*Privacy-Preserving Clinical Decision Support*")
    st.divider()
    
    # System Status
    st.markdown("### ⚙️ System Status")
    
    # Check Ollama
    ollama_ok = check_ollama_status()
    if ollama_ok:
        st.markdown("✅ **Ollama:** Running")
        models = list_models()
        if models:
            selected_model = st.selectbox(
                "Model",
                models,
                index=models.index(settings.OLLAMA_MODEL) if settings.OLLAMA_MODEL in models else 0
            )
        else:
            st.warning("No models found. Run: `ollama pull llama3`")
            selected_model = settings.OLLAMA_MODEL
    else:
        st.markdown("❌ **Ollama:** Not running")
        st.caption("Start with: `ollama serve`")
        selected_model = settings.OLLAMA_MODEL
    
    # Check Database
    doc_count = get_document_count()
    sources = get_unique_sources()
    
    if doc_count > 0:
        st.markdown("✅ **Database:** Connected")
        render_knowledge_base_stats(doc_count, sources)
    else:
        st.markdown("⚠️ **Database:** No documents")
        st.caption("Run: `python scripts/ingest_documents.py`")
    
    st.divider()
    
    # Settings
    st.markdown("### 🎛️ Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    top_k = st.slider(
        "Sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to use as context"
    )
    
    source_filter = st.selectbox(
        "Filter by document",
        ["All documents"] + sources,
        help="Optionally restrict search to specific document"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat_history()
        st.session_state.last_sources = []
        st.rerun()
    
    st.divider()
    st.caption("🔒 All processing happens locally")
    st.caption("No data leaves your machine")


# === Main Content ===
st.markdown('<p class="main-header">🏥 Clinical Decision Support System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about clinical guidelines and variant interpretation</p>', unsafe_allow_html=True)
st.divider()

# Check system readiness
system_ready = ollama_ok and doc_count > 0

if not system_ready:
    st.warning("⚠️ System not fully ready. Please check the sidebar for status.")
    
    if not ollama_ok:
        st.error("**Ollama is not running**. Start it with: `ollama serve`")
    
    if doc_count == 0:
        st.error("**No documents ingested**. Run: `python scripts/ingest_documents.py`")
    
    st.info("Once setup is complete, you can ask questions like:")
    st.markdown("""
    - *What are the ACMG criteria for pathogenic variants?*
    - *How should I interpret a BRCA1 mutation?*
    - *What evidence supports variant classification?*
    """)

# Render chat history
render_chat_history()

# Chat input
if prompt := render_chat_input():
    if not system_ready:
        st.error("Please set up the system first (see sidebar)")
    else:
        # Add user message
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    # Initialize pipeline if needed
                    pipeline = CDSSPipeline(
                        top_k=top_k,
                        temperature=temperature
                    )
                    
                    # Apply source filter
                    filter_source = None if source_filter == "All documents" else source_filter
                    
                    # Query the pipeline
                    result = pipeline.query(prompt, source_filter=filter_source)
                    
                    # Display answer
                    st.markdown(result.answer)
                    
                    # Store and display sources
                    st.session_state.last_sources = result.sources
                    if result.sources:
                        render_sources(result.sources)
                    
                    # Add to chat history
                    add_message("assistant", result.answer, result.sources)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    add_message("assistant", error_msg)


# === Footer ===
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔬 Powered by RAG + Local LLM")
with col2:
    st.caption(f"📚 {doc_count} chunks indexed")
with col3:
    st.caption("🏥 For clinical decision support only")
