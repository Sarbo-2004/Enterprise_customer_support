"""
FastAPI Backend
===============
Imports pipeline from pipeline.py — no duplication
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag_pipeline import run_pipeline, faiss_store   # ← import directly

app = FastAPI(title="Customer Support Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = "default"

class PipelineResponse(BaseModel):
    query:          str
    final_response: str
    agent_outputs:  list

@app.get("/")
def root():
    return {"status": "Customer Support API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "faiss_loaded": faiss_store is not None}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        final_response = run_pipeline(request.query)

        return {
            "query": request.query,
            "final_response": str(final_response),  # ✅ force string
            "agent_outputs": []
        }

    except Exception as e:
        return {
            "query": request.query,
            "final_response": f"Error: {str(e)}",
            "agent_outputs": []
        }