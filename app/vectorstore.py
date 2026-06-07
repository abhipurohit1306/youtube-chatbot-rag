import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vector_store


def save_vectorstore(vector_store, path):
    vector_store.save_local(path)

def load_vectorstore(path):
    embeddings = get_embeddings()
    return FAISS.load_local(path,embeddings,allow_dangerous_deserialization=True)


def get_or_create_vectorstore(video_id,get_chunks_func):
    path = f"faiss_indexes/{video_id}"
    if os.path.exists(path):
        print("Loading existing FAISS index...")
        return load_vectorstore(path)
    print("Creating new FAISS index...")
    chunks = get_chunks_func()
    vector_store = create_vectorstore(chunks)
    save_vectorstore(
        vector_store,
        path
    )
    return vector_store