# backend/app/deps.py
import os, openai, pinecone
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ.get("PINECONE_ENVIRONMENT"))
pinecone_index = pinecone.Index(os.environ.get("PINECONE_INDEX","rag-index"))

class OpenAIClient:
    def __init__(self):
        self._client = openai
    @property
    def embeddings(self):
        return self._client.Embedding
    @property
    def chat(self):
        return self._client.ChatCompletion

openai_client = OpenAIClient()
