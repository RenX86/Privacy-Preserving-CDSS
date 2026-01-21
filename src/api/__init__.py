"""API module - exports Ollama client and prompts"""
from .ollama_client import generate_response, generate_response_stream, check_ollama_status, list_models
from .prompts import SYSTEM_PROMPT, build_rag_prompt

__all__ = [
    "generate_response",
    "generate_response_stream",
    "check_ollama_status",
    "list_models",
    "SYSTEM_PROMPT",
    "build_rag_prompt"
]
