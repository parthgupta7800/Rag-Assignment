"""
Script to run the FastAPI server.
"""
import uvicorn
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

if __name__ == "__main__":
    print(f"Starting RAG Chatbot API server...")
    print(f"Host: {settings.HOST}")
    print(f"Port: {settings.PORT}")
    print(f"Debug: {settings.DEBUG}")
    print(f"NEC Pages Limit: {settings.NEC_MAX_PAGES}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
