from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# --- 2. CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post('/askowasp')
def ask_owasp_endpoint(req: QueryRequest):
    print(f"DEBUG: Received OWASP query: {req.query}")
    # Lazy import avoids heavy model loading during app startup.
    from owasp_chain import owasp_print

    answer = owasp_print(req.query)
    
    print(f"DEBUG: OWASP Answer: {answer}") 
    
    # --- 3. FIX: Return a Dictionary (JSON), not just the string ---
    return {'answer': answer} 

@app.post('/askmitre')
def ask_mitre_endpoint(req: QueryRequest):
    print(f"DEBUG: Received MITRE query: {req.query}")

    from mitre_chain import router
    answer = router.solve(req.query)
    
    print(f"DEBUG: MITRE Answer: {answer}")

    return {'answer': answer}

# To run: uvicorn run:app --reload
