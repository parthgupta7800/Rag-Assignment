# backend/app/ingest.py
import os, uuid
from .deps import openai_client, pinecone_index
from pdfminer.high_level import extract_text

def chunk_text(text, max_chars=2500, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks

def ingest_pdf_bytes(file_bytes, filename, source_label):
    text = extract_text(filename)  # for production: save bytes to file then extract page-wise
    chunks = chunk_text(text)
    embeddings = []
    for ch in chunks:
        emb = openai_client.embeddings.create(model="text-embedding-3-small", input=ch)["data"][0]["embedding"]
        meta = {"source": source_label, "doc_id": filename, "chunk_id": str(uuid.uuid4()), "text": ch[:1000]}
        pinecone_index.upsert([(meta["chunk_id"], emb, meta)])
