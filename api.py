# api.py (simplified)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_llama.config import AppConfig
from rag_llama.query_engine import RagChatEngine

app = FastAPI(title="ag-core RAG API", version="0.1.0")

# --- minimal request model ---
class QueryRequest(BaseModel):
    question: str

# --- lazy single engine ---
_engine = None

def get_engine() -> RagChatEngine:
    global _engine
    if _engine is None:
        cfg = AppConfig()
        _engine = RagChatEngine(cfg)
    return _engine

# --- routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        engine = get_engine()
        answer = engine.chat(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


