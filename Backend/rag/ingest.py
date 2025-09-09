# backend/rag/ingest.py
import os, glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def build_or_load_vs(data_dir="../data", persist_dir="../vectorstore"):
    os.makedirs(persist_dir, exist_ok=True)
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(collection_name="fitness", embedding_function=embed, persist_directory=persist_dir)
    if vs._collection.count() == 0:
        files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        docs=[]
        for f in files:
            for d in TextLoader(f, autodetect_encoding=True).load():
                docs.append(d)
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
        vs.add_documents(chunks); vs.persist()
    return vs
