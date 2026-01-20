"""
Settings Module - Centralized Configuration

PURPOSE:
This file loads all configuration from your .env file and makes it available
throughout your application. Instead of reading .env in every file, you 
just import 'settings' from here.

USAGE:
    from src.config.settings import settings
    
    # Access database settings
    print(settings.POSTGRES_HOST)
    print(settings.POSTGRES_PORT)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root (where .env lives)
# This file is at: src/config/settings.py
# So we go up 2 levels to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env file
load_dotenv(ENV_PATH)


class Settings:
    """
    Configuration settings loaded from environment variables.
    
    All your secrets and configuration live in .env file.
    This class reads them and provides easy access.
    """
    
    # === Database Settings ===
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "cdss-vectorbase")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    
    # === Embedding Model Settings ===
    # Using sentence-transformers model - this one is small and fast
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # === Ollama (Local LLM) Settings ===
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    
    # === Document Processing Settings ===
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))  # words per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # overlap words
    
    # === RAG Settings ===
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))  # documents to retrieve
    
    # === Paths ===
    DATA_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed"
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def model_dump(self) -> dict:
        """Return all settings as a dictionary (useful for debugging)"""
        return {
            "POSTGRES_USER": self.POSTGRES_USER,
            "POSTGRES_DB": self.POSTGRES_DB,
            "POSTGRES_HOST": self.POSTGRES_HOST,
            "POSTGRES_PORT": self.POSTGRES_PORT,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "OLLAMA_HOST": self.OLLAMA_HOST,
            "OLLAMA_MODEL": self.OLLAMA_MODEL,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "TOP_K_RESULTS": self.TOP_K_RESULTS,
        }


# Create a single instance to use throughout the app
settings = Settings()


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== CDSS Settings ===")
    for key, value in settings.model_dump().items():
        print(f"  {key}: {value}")
    print(f"\n  DATABASE_URL: {settings.DATABASE_URL}")
