"""
RAG (Retrieval-Augmented Generation) service that combines document retrieval with LLM generation.
"""
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .gemini_service import gemini_service
from .vector_store import vector_store
from .document_processor import document_processor
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Service for handling RAG pipeline operations."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.gemini = gemini_service
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.conversation_memory = {}  # Simple in-memory storage for conversations
        logger.info("Initialized RAG service")
    
    def ingest_document(
        self, 
        file_content: bytes, 
        filename: str, 
        source: str,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_content: File bytes
            filename: Name of the file
            source: Source category (NEC, WATTMONK, etc.)
            additional_metadata: Additional metadata
            
        Returns:
            Ingestion result with statistics
        """
        try:
            logger.info(f"Starting document ingestion: {filename} (source: {source})")
            
            # Process document into chunks
            chunks = self.document_processor.process_document(
                file_content, filename, source, additional_metadata
            )
            
            # Generate embeddings for chunks
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.gemini.generate_embeddings(texts)
            
            # Store in vector database
            document_ids = self.vector_store.add_documents(chunks, embeddings, source)
            
            result = {
                "status": "success",
                "filename": filename,
                "source": source,
                "chunks_created": len(chunks),
                "document_ids": document_ids,
                "total_characters": sum(len(chunk["content"]) for chunk in chunks)
            }
            
            logger.info(f"Successfully ingested {filename}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document {filename}: {str(e)}")
            raise
    
    def query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        source_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a query using the RAG pipeline.
        
        Args:
            query: User's question
            session_id: Session identifier for conversation memory
            source_filter: Specific source to search (optional)
            top_k: Number of context documents to retrieve
            
        Returns:
            Response with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Generate session_id if not provided
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())

            # Classify intent if no source filter provided
            if not source_filter:
                intent = self.gemini.classify_intent(query)
                source_filter = intent if intent in settings.DOCUMENT_SOURCES else None

            # Generate query embedding
            query_embedding = self.gemini.generate_query_embedding(query)

            # Retrieve relevant documents
            relevant_docs = self.vector_store.search_similar(
                query_embedding,
                source=source_filter,
                top_k=top_k or settings.TOP_K_RESULTS
            )

            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)

            # Get conversation history
            conversation_history = self._get_conversation_history(session_id)
            
            # Generate response
            response = self.gemini.generate_response(query, context, conversation_history)
            
            # Update conversation memory
            if session_id:
                self._update_conversation_memory(session_id, query, response)
            
            # Prepare result
            result = {
                "answer": response,
                "sources": self._format_sources(relevant_docs),
                "context_used": len(relevant_docs),
                "intent_classification": source_filter or "GENERAL",
                "confidence_score": self._calculate_confidence_score(relevant_docs),
                "session_id": session_id
            }
            
            logger.info(f"Generated response with {len(relevant_docs)} context documents")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from retrieved documents.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Formatted context string
        """
        if not relevant_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc["metadata"]
            source = metadata.get("source", "Unknown")
            filename = metadata.get("filename", "Unknown")
            score = doc.get("score", 0)
            
            context_parts.append(
                f"[Source {i}: {source} - {filename} (Relevance: {score:.2f})]\n"
                f"{doc['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source information for the response.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Formatted source information
        """
        sources = []
        for doc in relevant_docs:
            metadata = doc["metadata"]
            sources.append({
                "source": metadata.get("source", "Unknown"),
                "filename": metadata.get("filename", "Unknown"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "relevance_score": doc.get("score", 0),
                "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            })
        
        return sources
    
    def _calculate_confidence_score(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not relevant_docs:
            return 0.0
        
        # Simple confidence calculation based on top document score and number of docs
        top_score = relevant_docs[0].get("score", 0) if relevant_docs else 0
        doc_count_factor = min(len(relevant_docs) / settings.TOP_K_RESULTS, 1.0)
        
        confidence = (top_score * 0.7) + (doc_count_factor * 0.3)
        return round(confidence, 2)
    
    def _get_conversation_history(self, session_id: Optional[str]) -> Optional[List[Dict[str, str]]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation history or None
        """
        if not session_id or session_id not in self.conversation_memory:
            return None
        
        return self.conversation_memory[session_id]
    
    def _update_conversation_memory(self, session_id: str, query: str, response: str):
        """
        Update conversation memory with new query and response.
        
        Args:
            session_id: Session identifier
            query: User's query
            response: Assistant's response
        """
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        conversation = self.conversation_memory[session_id]
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": response})
        
        # Keep only last 10 messages to manage memory
        if len(conversation) > 10:
            self.conversation_memory[session_id] = conversation[-10:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and health information.
        
        Returns:
            System statistics
        """
        try:
            collection_stats = self.vector_store.get_collection_stats()
            
            stats = {
                "status": "healthy",
                "collections": collection_stats,
                "active_sessions": len(self.conversation_memory),
                "total_documents": sum(stats["document_count"] for stats in collection_stats.values()),
                "configuration": {
                    "chunk_size": settings.CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "top_k_results": settings.TOP_K_RESULTS,
                    "gemini_model": settings.GEMINI_MODEL
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"status": "error", "error": str(e)}

# Create global instance
rag_service = RAGService()
