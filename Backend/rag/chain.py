# Backend/rag/chain.py
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

PERSIST_DIR = Path(__file__).resolve().parents[1] / "vectorstore"

SYSTEM = (
    "You are a concise, safe fitness coach. "
    "Answer ONLY with the retrieved context. If the context is empty or irrelevant, say "
    "'I don't have that in my knowledge base.' "
    "No medical or injury advice. "
    "Prefer actionable guidance. Min 5 bullet points; use sets x reps x rest when relevant."
)

HUMAN = (
    "User question: {question}\n\n"
    "Use ONLY this context:\n{context}"
)

def build_retriever(k: int = 4):
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(
        collection_name="fitness",
        embedding_function=embed,
        persist_directory=str(PERSIST_DIR),
    )
    return vs.as_retriever(search_kwargs={"k": k})

def build_qa(llm):
    retriever = build_retriever(k=4)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", HUMAN),
    ])
    # pass prompt into the chain so the LLM must use {context}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  # set True if you want citations
    )
    return qa
