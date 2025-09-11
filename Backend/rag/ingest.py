# backend/rag/ingest.py
from pathlib import Path
from typing import List
import os, glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DATA_DIR = Path(__file__).resolve().parents[1]/"data"
PERSIST_DIR = Path(__file__).resolve().parents[1]/"vectorstore"

def load_docs() -> List:
    docs = []
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in (".txt", ".md"):
            docs += TextLoader(str(p), autodetect_encoding=True).load()
        elif p.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(p)).load()
    return docs

def build_or_update_index():
    DATA_DIR.mkdir(exist_ok=True)
    PERSIST_DIR.mkdir(parents=True,exist_ok=True)

    docs = load_docs()
    if not docs:
        raise RuntimeError(f"No files in {DATA_DIR}. Add .txt/.md/.pdf first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(
        collection_name="fitness",
        embedding_function=embed,
        persist_directory=str(PERSIST_DIR),
    )
    # upsert
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)


