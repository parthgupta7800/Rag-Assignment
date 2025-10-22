"""
Configuration settings for the RAG Chatbot application.
"""
import os
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    STREAMLIT_PORT: int = 8501
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    MAX_CONVERSATION_HISTORY: int = 10
    
    # Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # Document Sources
    DOCUMENT_SOURCES: Dict[str, str] = {
        "NEC": "NEC Code Guidelines",
        "WATTMONK": "Wattmonk Company Information", 
        "GENERAL": "General Knowledge"
    }
    
    # NEC Processing Configuration
    NEC_MAX_PAGES: int = 100  # Process only first 100 pages of NEC book
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Validate required settings
if not settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
