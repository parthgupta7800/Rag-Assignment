"""
Script to ingest the provided NEC and Wattmonk documents into the RAG system.
Only processes the first 100 pages of the NEC book.
"""
import os
import sys
import asyncio
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_service import rag_service
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_document_file(file_path: str, source: str):
    """
    Ingest a single document file.
    
    Args:
        file_path: Path to the document file
        source: Source category (NEC, WATTMONK, etc.)
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        logger.info(f"Ingesting {file_path.name} as {source}...")
        
        # Special handling for NEC book
        if source == "NEC":
            logger.info(f"NEC document detected - will process only first {settings.NEC_MAX_PAGES} pages")
        
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Ingest document
        result = rag_service.ingest_document(
            file_content=file_content,
            filename=file_path.name,
            source=source,
            additional_metadata={
                "file_size": len(file_content),
                "file_path": str(file_path),
                "max_pages_processed": settings.NEC_MAX_PAGES if source == "NEC" else "all"
            }
        )
        
        logger.info(f"Successfully ingested {file_path.name}:")
        logger.info(f"  - Chunks created: {result['chunks_created']}")
        logger.info(f"  - Total characters: {result['total_characters']}")
        
        if source == "NEC":
            logger.info(f"  - Pages processed: First {settings.NEC_MAX_PAGES} pages only")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting {file_path}: {str(e)}")
        return False

async def main():
    """Main function to ingest all provided documents."""
    logger.info("Starting document ingestion process...")
    logger.info(f"NEC processing limit: {settings.NEC_MAX_PAGES} pages")
    
    # Define documents to ingest
    documents = [
        {
            "path": "../data/2017-NEC-Code-2 (2) (1).pdf",
            "source": "NEC",
            "description": f"2017 NEC Code Guidelines (First {settings.NEC_MAX_PAGES} pages)"
        },
        {
            "path": "../data/Wattmonk Information (1) (1).docx",
            "source": "WATTMONK",
            "description": "Wattmonk Company Information"
        }
    ]
    
    # Check if documents exist
    existing_documents = []
    for doc in documents:
        if os.path.exists(doc["path"]):
            existing_documents.append(doc)
            logger.info(f"Found document: {doc['path']}")
        else:
            logger.warning(f"Document not found: {doc['path']}")
    
    if not existing_documents:
        logger.error("No documents found to ingest!")
        logger.info("Please ensure the following files exist:")
        for doc in documents:
            logger.info(f"  - {doc['path']}")
        return
    
    # Ingest documents
    success_count = 0
    for doc in existing_documents:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {doc['description']}")
        logger.info(f"File: {doc['path']}")
        logger.info(f"Source: {doc['source']}")
        logger.info(f"{'='*50}")
        
        success = await ingest_document_file(doc["path"], doc["source"])
        if success:
            success_count += 1
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INGESTION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total documents processed: {len(existing_documents)}")
    logger.info(f"Successfully ingested: {success_count}")
    logger.info(f"Failed: {len(existing_documents) - success_count}")
    logger.info(f"NEC pages limit: {settings.NEC_MAX_PAGES}")
    
    if success_count > 0:
        logger.info("\n✅ Document ingestion completed successfully!")
        logger.info("You can now start the Streamlit app to interact with the chatbot.")
        logger.info("\nTo start the application:")
        logger.info("1. Start the API server: python api/main.py")
        logger.info("2. Start the Streamlit app: streamlit run streamlit_app.py")
    else:
        logger.error("\n❌ Document ingestion failed!")
    
    # Display system stats
    try:
        stats = rag_service.get_system_stats()
        logger.info(f"\nSystem Statistics:")
        logger.info(f"Total documents in database: {stats['total_documents']}")
        for source, collection_stats in stats['collections'].items():
            logger.info(f"  - {source}: {collection_stats['document_count']} documents")
    except Exception as e:
        logger.warning(f"Could not retrieve system stats: {str(e)}")

if __name__ == "__main__":
    # Check if required environment variables are set
    if not settings.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable is not set!")
        logger.info("Please set your Google API key in a .env file or environment variable.")
        logger.info("Example: GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run the ingestion process
    asyncio.run(main())
