"""
Script to run the Streamlit app.
"""
import subprocess
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

if __name__ == "__main__":
    print(f"Starting RAG Chatbot Streamlit app...")
    print(f"Port: {settings.STREAMLIT_PORT}")
    print(f"NEC Pages Limit: {settings.NEC_MAX_PAGES}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", "0.0.0.0"
    ]
    
    subprocess.run(cmd)
