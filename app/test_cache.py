from ingestion import get_video_chunks

from vectorstore import (
    get_or_create_vectorstore
)

video_id = "oAkLSJNr5zY"

chunks = get_video_chunks(
    video_id
)

vector_store = get_or_create_vectorstore(
    video_id,
    chunks
)

print("Ready")