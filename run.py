from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. REAL IMPORTS (Uncommented) ---
# Ensure these files (owasp_chain.py, mitre_chain.py) are in the same folder
from owasp_chain import owasp_print
from mitre_chain import router 

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
    
    # Call the real function imported from owasp_chain.py
    answer = owasp_print(req.query)
    
    print(f"DEBUG: OWASP Answer: {answer}") 
    
    # --- 3. FIX: Return a Dictionary (JSON), not just the string ---
    return {'answer': answer} 

@app.post('/askmitre')
def ask_mitre_endpoint(req: QueryRequest):
    print(f"DEBUG: Received MITRE query: {req.query}")

    # Call the real function imported from mitre_chain.py
    answer = router.solve(req.query)
    
    print(f"DEBUG: MITRE Answer: {answer}")

    return {'answer': answer}

# To run: uvicorn backend:app --reload