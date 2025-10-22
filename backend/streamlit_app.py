"""
Streamlit frontend for the RAG Chatbot application.
"""
import streamlit as st
import requests
import json
import uuid
from typing import Dict, Any, List
import time
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = f"http://localhost:{settings.PORT}"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_health" not in st.session_state:
    st.session_state.api_health = None

# Helper functions
def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_stats():
    """Get system statistics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def send_query(query: str, source_filter: str = None, top_k: int = 5):
    """Send a query to the RAG API."""
    try:
        payload = {
            "query": query,
            "session_id": st.session_state.session_id,
            "top_k": top_k
        }
        
        if source_filter and source_filter != "AUTO":
            payload["source_filter"] = source_filter
        
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def upload_document(file, source: str):
    """Upload a document to the RAG system."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"source": source}
        
        response = requests.post(
            f"{API_BASE_URL}/api/ingest",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Upload Error: {str(e)}"}

# Main app
def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("*Retrieval-Augmented Generation Chatbot with Multi-Context Support*")
    st.markdown("*Using Gemini 2.0 Flash and Chroma DB*")
    
    # Check API health
    if st.session_state.api_health is None:
        st.session_state.api_health = check_api_health()
    
    if not st.session_state.api_health:
        st.error("⚠️ API is not running. Please start the FastAPI server first.")
        st.code(f"cd backend && python api/main.py")
        if st.button("Retry Connection"):
            st.session_state.api_health = check_api_health()
            st.rerun()
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Source filter
        source_options = ["AUTO", "NEC", "WATTMONK", "GENERAL"]
        source_filter = st.selectbox(
            "Context Source",
            source_options,
            help="Choose specific context or AUTO for intelligent routing"
        )
        
        # Number of context documents
        top_k = st.slider(
            "Context Documents",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant documents to retrieve"
        )
        
        st.divider()
        
        # System stats
        st.header("📊 System Stats")
        if st.button("Refresh Stats"):
            stats = get_system_stats()
            if stats:
                st.json(stats)
            else:
                st.error("Failed to get stats")
        
        st.divider()
        
        # Document upload
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            help="Upload PDF or DOCX files to add to the knowledge base"
        )
        
        if uploaded_file:
            upload_source = st.selectbox(
                "Document Source",
                ["NEC", "WATTMONK", "GENERAL"],
                help="Categorize the document"
            )
            
            if st.button("Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    result = upload_document(uploaded_file, upload_source)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"✅ Document uploaded successfully!")
                        st.json(result)
        
        st.divider()
        
        # Session management
        st.header("💬 Session")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        
        if st.button("New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Configuration info
        st.header("⚙️ Configuration")
        st.text(f"NEC Pages Limit: {settings.NEC_MAX_PAGES}")
        st.text(f"Chunk Size: {settings.CHUNK_SIZE}")
        st.text(f"Model: {settings.GEMINI_MODEL}")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']} - {source['filename']}")
                        st.markdown(f"*Relevance: {source['relevance_score']:.2f}*")
                        st.markdown(f"```\n{source['preview']}\n```")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about NEC codes, Wattmonk, or general topics..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_query(
                    prompt, 
                    source_filter if source_filter != "AUTO" else None,
                    top_k
                )
                
                if "error" in response:
                    st.error(response["error"])
                    return
                
                # Display response
                st.markdown(response["answer"])
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent", response["intent_classification"])
                with col2:
                    st.metric("Confidence", f"{response['confidence_score']:.2f}")
                with col3:
                    st.metric("Context Docs", response["context_used"])
                
                # Store assistant message with sources
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "metadata": {
                        "intent": response["intent_classification"],
                        "confidence": response["confidence_score"],
                        "context_used": response["context_used"]
                    }
                }
                st.session_state.messages.append(assistant_message)

# Example queries section
def show_example_queries():
    st.header("💡 Example Queries")
    
    examples = {
        "NEC Code Questions (First 100 pages)": [
            "What are the requirements for GFCI protection?",
            "What is the minimum wire gauge for circuits?",
            "What are electrical safety requirements?"
        ],
        "Wattmonk Questions": [
            "What services does Wattmonk provide?",
            "How does Wattmonk handle solar design projects?",
            "What is Wattmonk's pricing structure?"
        ],
        "General Questions": [
            "How do solar panels work?",
            "What is the difference between AC and DC power?",
            "Explain electrical load calculations"
        ]
    }
    
    for category, queries in examples.items():
        with st.expander(category):
            for query in queries:
                if st.button(query, key=f"example_{hash(query)}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()
    
    # Show examples at the bottom
    if not st.session_state.messages:
        show_example_queries()
