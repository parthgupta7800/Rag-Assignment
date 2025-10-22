"""
Chroma vector database service for storing and retrieving document embeddings.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
import uuid
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Service for managing document embeddings in Chroma DB."""
    
    def __init__(self):
        """Initialize the Chroma vector store."""
        try:
            # Initialize Chroma client with persistent storage
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collections for different document sources
            self.collections = {}
            for source_key, source_name in settings.DOCUMENT_SOURCES.items():
                collection_name = f"documents_{source_key.lower()}"
                self.collections[source_key] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": f"Collection for {source_name}"}
                )
            
            logger.info(f"Initialized Chroma vector store with {len(self.collections)} collections")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        source: str
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of document chunks with metadata
            embeddings: List of embedding vectors
            source: Source category (NEC, WATTMONK, GENERAL)
            
        Returns:
            List of document IDs
        """
        try:
            if source not in self.collections:
                raise ValueError(f"Unknown source: {source}")
            
            collection = self.collections[source]
            
            # Prepare data for Chroma
            ids = []
            texts = []
            metadatas = []
            
            for doc, embedding in zip(documents, embeddings):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc["content"])
                
                # Prepare metadata (Chroma requires string values)
                metadata = {}
                for key, value in doc["metadata"].items():
                    metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to {source} collection")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        source: Optional[str] = None, 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            source: Specific source to search (optional)
            top_k: Number of results to return
            
        Returns:
            List of similar documents with metadata and scores
        """
        try:
            if top_k is None:
                top_k = settings.TOP_K_RESULTS
            
            all_results = []
            
            # Determine which collections to search
            collections_to_search = {}
            if source and source in self.collections:
                collections_to_search[source] = self.collections[source]
            else:
                collections_to_search = self.collections
            
            # Search each collection
            for source_key, collection in collections_to_search.items():
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Process results
                    if results["documents"] and results["documents"][0]:
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results["documents"][0],
                            results["metadatas"][0],
                            results["distances"][0]
                        )):
                            all_results.append({
                                "content": doc,
                                "metadata": {
                                    **metadata,
                                    "source": source_key,
                                    "similarity_score": 1 - distance  # Convert distance to similarity
                                },
                                "score": 1 - distance
                            })
                
                except Exception as e:
                    logger.warning(f"Error searching collection {source_key}: {str(e)}")
                    continue
            
            # Sort by similarity score and return top results
            all_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = all_results[:top_k]
            
            logger.info(f"Found {len(final_results)} similar documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {}
            for source_key, collection in self.collections.items():
                count = collection.count()
                stats[source_key] = {
                    "document_count": count,
                    "collection_name": collection.name
                }
            
            logger.info("Retrieved collection statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise
    
    def delete_collection(self, source: str) -> bool:
        """
        Delete a collection and all its documents.
        
        Args:
            source: Source category to delete
            
        Returns:
            True if successful
        """
        try:
            if source not in self.collections:
                raise ValueError(f"Unknown source: {source}")
            
            collection_name = self.collections[source].name
            self.client.delete_collection(collection_name)
            
            # Recreate the collection
            self.collections[source] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Collection for {settings.DOCUMENT_SOURCES[source]}"}
            )
            
            logger.info(f"Deleted and recreated collection for source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_all_collections(self) -> bool:
        """
        Reset all collections (delete all data).
        
        Returns:
            True if successful
        """
        try:
            for source in self.collections.keys():
                self.delete_collection(source)
            
            logger.info("Reset all collections")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collections: {str(e)}")
            raise

# Create global instance
vector_store = VectorStore()
