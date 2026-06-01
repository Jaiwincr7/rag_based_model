import asyncio
import logging
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    from rag_components import hf_token_configured

    owasp_db = os.path.isdir("./chroma_db/owasp")
    mitre_db = os.path.isdir("./chroma_db/mitre_attack_v5")

    return {
        "status": "ok",
        "hf_token_configured": hf_token_configured(),
        "owasp_db_present": owasp_db,
        "mitre_db_present": mitre_db,
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
                "Retry once; HF inference can be slow on cold start."
            ),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("askowasp failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("askmitre failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer}
