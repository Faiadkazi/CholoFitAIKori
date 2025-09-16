# Backend/app.
from fastapi import FastAPI, HTTPException
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


qa = None


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/api/health", "/api/ingest", "/api/chat"]}

@app.get("/api/health")
def health():
    return {"ok": True, "rag is ready": qa is not None}

@app.post("/api/ingest")
def ingest():
    from Backend.rag.ingest import build_or_update_index
    from Backend.rag.chain import build_qa
    try:
        n = build_or_update_index()
        global qa
        qa = build_qa(llm)  # reload retriever
        return {"ok": True, "chunks_indexed": n}
    except Exception as e:
        raise HTTPException(400, f"Ingest failed: {e}")


@app.post("/api/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    global qa, llm
    if qa is None:

        resp = llm.invoke(inp.message)
        return ChatOut(reply=getattr(resp, "content", str(resp)))
    out = qa.invoke({"query":inp.message})
    result = out.get("result") if isinstance(out,dict) else str(out).strip()
    srcs = (out.get("source_documents") or []) if isinstance(out,dict) else[]
    tail = ""
    if not result or "I don't have that in my knowledge base" in result:
        m = llm.invoke(inp.message)
        result = getattr(m, "content",str(m))
        return ChatOut(reply=result)
    if srcs:
        # show top 1-2 source snippets for verification
        snips = []
        for d in srcs[:2]:
            snips.append((d.metadata.get("source","?"), d.page_content[:180].replace("\n"," ")))
        tail = "\n\n[SOURCES]\n" + "\n".join(f"- {s}: {t}" for s,t in snips)
    return ChatOut(reply=(result or ""))

@app.on_event("startup")
def startup_event():
    from Backend.rag.ingest import build_or_update_index
    from Backend.rag.chain import build_qa
    global qa
    try:
        n = build_or_update_index()
        qa = build_qa(llm)
        print(f" RAG index loaded with {n} chunks")
    except Exception as e:
        print(f" Failed to build RAG index at startup: {e}")
        qa = None

