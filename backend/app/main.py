# backend/app/main.py
import os
from fastapi import FastAPI
from . import rag

app = FastAPI(title="RAG Chatbot")

@app.get("/")
async def root():
    return {"status": "ok", "service": "RAG Chatbot"}

# import routes from rag
app.include_router(rag.router, prefix="/api")
