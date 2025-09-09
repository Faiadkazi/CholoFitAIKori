# Backend/app.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_ollama import ChatOllama


app = FastAPI(title="CholoFitAI")

# allow your Live Server origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    user_id: Optional[str] = None
    message: str

class ChatOut(BaseModel):
    reply: str

llm = ChatOllama(base_url="http://localhost:11434", 
                 model= "llama3.2:3b-instruct-q4_K_M",
                 temperature=0.3,)



@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    resp = llm.invoke(inp.message)
    return ChatOut(reply=getattr(resp, "content", str(resp)))


