import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "90"))

app = FastAPI(title="Security RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    """Fast health check — Render should use this path, not /askowasp."""
    from rag_components import hf_token_configured

    return {
        "status": "ok",
        "hf_token_configured": hf_token_configured(),
    }


@app.post("/askowasp")
async def ask_owasp_endpoint(req: QueryRequest):
    from owasp_chain import owasp_print

    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(owasp_print, req.query),
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"OWASP request timed out after {REQUEST_TIMEOUT_SEC}s. "
                "Cold starts load embeddings + call Hugging Face — retry once."
            ),
        )
    return {"answer": answer}


@app.post("/askmitre")
async def ask_mitre_endpoint(req: QueryRequest):
    from mitre_chain import router

    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(router.solve, req.query),
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"MITRE request timed out after {REQUEST_TIMEOUT_SEC}s.",
        )
    return {"answer": answer}
