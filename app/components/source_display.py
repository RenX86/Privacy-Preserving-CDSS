"""
Source Display Component

PURPOSE:
Display source documents and citations in a structured way.
Shows which guidelines were used to generate responses.

USAGE:
    from app.components.source_display import render_sources, render_source_panel
"""

import streamlit as st
from typing import List, Dict, Any


def render_sources(sources: List[Dict[str, Any]], expanded: bool = False):
    """
    Render source citations in an expander.
    
    Args:
        sources: List of source dictionaries
        expanded: Whether the expander is open by default
    """
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} references)", expanded=expanded):
        for i, src in enumerate(sources, 1):
            render_source_card(src, i)


def render_source_card(source: Dict[str, Any], index: int):
    """Render a single source as a card."""
    similarity = source.get('similarity', 0)
    
    # Color based on relevance
    if similarity >= 0.8:
        color = "🟢"
    elif similarity >= 0.6:
        color = "🟡"
    else:
        color = "🔴"
    
    st.markdown(f"""
    **{index}. {source.get('source_file', 'Unknown Source')}** {color}
    
    *Relevance: {similarity:.1%}*
    """)
    
    if source.get('preview'):
        st.caption(f"_{source['preview']}_")
    
    st.divider()


def render_source_panel(sources: List[Dict[str, Any]]):
    """
    Render sources in the sidebar panel.
    
    For use in a sidebar layout.
    """
    st.sidebar.markdown("### 📚 Sources Used")
    
    if not sources:
        st.sidebar.info("No sources available yet. Ask a question!")
        return
    
    for i, src in enumerate(sources, 1):
        similarity = src.get('similarity', 0)
        
        # Relevance indicator
        if similarity >= 0.8:
            indicator = "🟢 High"
        elif similarity >= 0.6:
            indicator = "🟡 Medium"
        else:
            indicator = "🔴 Low"
        
        with st.sidebar.expander(f"{i}. {src.get('source_file', 'Unknown')[:20]}..."):
            st.markdown(f"**Relevance:** {indicator} ({similarity:.1%})")
            st.markdown(f"**Chunk:** {src.get('chunk_index', 'N/A')}")
            if src.get('preview'):
                st.caption(src['preview'])


def render_knowledge_base_stats(doc_count: int, sources: List[str]):
    """
    Render knowledge base statistics.
    
    Args:
        doc_count: Total document chunks in database
        sources: List of unique source files
    """
    st.sidebar.markdown("### 📊 Knowledge Base")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Chunks", doc_count)
    col2.metric("Documents", len(sources))
    
    if sources:
        with st.sidebar.expander("View Documents"):
            for src in sources:
                st.markdown(f"• {src}")
