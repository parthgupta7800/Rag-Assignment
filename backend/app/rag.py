# backend/app/rag.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    session_id: str | None = None
    query: str
    top_k: int = 5
    context: str | None = None  # "NEC", "Wattmonk", or None

@router.post("/query")
async def query_endpoint(req: QueryRequest):
    # placeholder: call intent classifier, retrieval, LLM
    return {"answer": "This is a placeholder — implement retrieval & LLM call."}

@router.post("/ingest")
async def ingest(file: UploadFile = File(...), source: str = Form(...)):
    # placeholder: save file & run ingestion
    return {"status": "received", "filename": file.filename, "source": source}
