"""
Document processing service for extracting and chunking text from various file formats.
"""
import PyPDF2
from docx import Document
from typing import List, Dict, Any
import logging
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing documents and extracting text."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("Initialized document processor")
    
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from PDF file content.
        
        Args:
            file_content: PDF file bytes
            filename: Name of the file
            
        Returns:
            Extracted text content
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            total_pages = len(pdf_reader.pages)

            # Check if this is the NEC book and limit to first 100 pages
            max_pages = total_pages
            if "nec" in filename.lower() or "code" in filename.lower():
                max_pages = min(settings.NEC_MAX_PAGES, total_pages)
                logger.info(f"Processing NEC document: limiting to first {max_pages} pages out of {total_pages}")

            for page_num, page in enumerate(pdf_reader.pages):
                # Stop if we've reached the page limit
                if page_num >= max_pages:
                    logger.info(f"Reached page limit ({max_pages}), stopping extraction")
                    break

                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {filename}: {str(e)}")
                    continue
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from PDF: {filename}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filename}: {str(e)}")
            raise
    
    def extract_text_from_docx(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from DOCX file content.
        
        Args:
            file_content: DOCX file bytes
            filename: Name of the file
            
        Returns:
            Extracted text content
        """
        try:
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from DOCX: {filename}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filename}: {str(e)}")
            raise
    
    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from file based on file extension.
        
        Args:
            file_content: File bytes
            filename: Name of the file
            
        Returns:
            Extracted text content
        """
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file_content, filename)
        elif file_extension in ['docx', 'doc']:
            return self.extract_text_from_docx(file_content, filename)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of text chunks with metadata
        """
        try:
            chunks = self.text_splitter.split_text(text)
            
            chunked_documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    **(metadata or {})
                }
                
                chunked_documents.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunked_documents)} chunks from text")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def process_document(
        self, 
        file_content: bytes, 
        filename: str, 
        source: str,
        additional_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a document: extract text and create chunks with metadata.
        
        Args:
            file_content: File bytes
            filename: Name of the file
            source: Source category (NEC, WATTMONK, etc.)
            additional_metadata: Additional metadata
            
        Returns:
            List of processed document chunks
        """
        try:
            # Extract text
            text = self.extract_text_from_file(file_content, filename)
            
            # Prepare metadata
            metadata = {
                "filename": filename,
                "source": source,
                "file_type": filename.lower().split('.')[-1],
                **(additional_metadata or {})
            }
            
            # Create chunks
            chunks = self.chunk_text(text, metadata)
            
            logger.info(f"Processed document {filename}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise

# Create global instance
document_processor = DocumentProcessor()
