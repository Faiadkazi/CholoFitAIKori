# Backend/rag/chain.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from .retriever import get_retriever
from .ingest import build_or_load_vs

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM = (
    "You are a concise, safe fitness coach. "
    "Use ONLY retrieved context. No medical, injury, or diet advice. "
    "Output at most 5 short steps with sets x reps x rest. "
)

def build_chain():
    # ensure vector store exists or is populated
    build_or_load_vs(data_dir="../data", persist_dir="../vectorstore")
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "User question: {question}\nAnswer using retrieved context.")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

