"""
FastAPI application for the RAG Chatbot.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rag_service import rag_service
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation Chatbot with Multi-Context Support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    source_filter: Optional[str] = None
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: int
    intent_classification: str
    confidence_score: float
    session_id: str

class IngestResponse(BaseModel):
    status: str
    filename: str
    source: str
    chunks_created: int
    document_ids: List[str]
    total_characters: int

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": "1.0.0"
    }

# Main query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Submit a query to the RAG chatbot.
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        result = rag_service.query(
            query=request.query,
            session_id=request.session_id,
            source_filter=request.source_filter,
            top_k=request.top_k or 5
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Document ingestion endpoint
@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    additional_metadata: Optional[str] = Form(None)
):
    """
    Upload and ingest a document into the knowledge base.
    """
    try:
        # Validate source
        if source not in settings.DOCUMENT_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {list(settings.DOCUMENT_SOURCES.keys())}"
            )
        
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and DOCX files are supported"
            )
        
        logger.info(f"Ingesting document: {file.filename} as {source}")
        
        # Read file content
        file_content = await file.read()
        
        # Parse additional metadata if provided
        metadata = {}
        if additional_metadata:
            try:
                import json
                metadata = json.loads(additional_metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid additional_metadata JSON, ignoring")
        
        # Ingest document
        result = rag_service.ingest_document(
            file_content=file_content,
            filename=file.filename,
            source=source,
            additional_metadata=metadata
        )
        
        return IngestResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

# System statistics endpoint
@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics and health information.
    """
    try:
        stats = rag_service.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

# Available sources endpoint
@app.get("/api/sources")
async def get_available_sources():
    """
    Get list of available document sources.
    """
    return {
        "sources": [
            {"key": key, "name": name}
            for key, name in settings.DOCUMENT_SOURCES.items()
        ]
    }

# Collection management endpoint
@app.delete("/api/collections/{source}")
async def reset_collection(source: str):
    """
    Reset a specific document collection.
    """
    try:
        if source not in settings.DOCUMENT_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {list(settings.DOCUMENT_SOURCES.keys())}"
            )
        
        # Reset the collection
        rag_service.vector_store.reset_collection(source)
        
        return {
            "status": "success",
            "message": f"Collection {source} reset successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting collection {source}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting collection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
