# backend/rag/retriever.py
from .ingest import build_or_load_vs
def get_retriever():
    vs = build_or_load_vs()
    return vs.as_retriever(search_kwargs={"k":4})
