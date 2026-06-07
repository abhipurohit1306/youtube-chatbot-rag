from ingestion import get_video_chunks
from vectorstore import (
    create_vectorstore,
    save_vectorstore,
    load_vectorstore
)

chunks = get_video_chunks(
    "oAkLSJNr5zY"
)

vector_store = create_vectorstore(
    chunks
)

save_vectorstore(
    vector_store,
    "faiss_indexes/test_video"
)

loaded_store = load_vectorstore(
    "faiss_indexes/test_video"
)

print("Loaded Successfully")