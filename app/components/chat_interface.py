"""
Chat Interface Component

PURPOSE:
Reusable chat interface component for the Streamlit app.
Handles message display, input, and chat history.

USAGE:
    from app.components.chat_interface import render_chat, add_message
"""

import streamlit as st
from typing import List, Dict, Any


def initialize_chat_history():
    """Initialize chat history in session state if not present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


def add_message(role: str, content: str, sources: List[Dict] = None):
    """
    Add a message to chat history.
    
    Args:
        role: 'user' or 'assistant'
        content: Message text
        sources: Optional list of source citations (for assistant messages)
    """
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear all messages from chat history."""
    st.session_state.messages = []


def render_message(message: Dict[str, Any]):
    """Render a single chat message."""
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Show sources for assistant messages
        if role == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Sources", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**{i}. {src.get('source_file', 'Unknown')}** "
                        f"(relevance: {src.get('similarity', 0):.2f})"
                    )
                    if src.get('preview'):
                        st.caption(src['preview'])


def render_chat_history():
    """Render all messages in chat history."""
    for message in st.session_state.messages:
        render_message(message)


def render_chat_input(placeholder: str = "Ask a clinical question...") -> str:
    """
    Render the chat input field.
    
    Returns:
        User input text or None
    """
    return st.chat_input(placeholder, disabled=st.session_state.processing)


def render_thinking_indicator():
    """Show a thinking/processing indicator."""
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            st.empty()
