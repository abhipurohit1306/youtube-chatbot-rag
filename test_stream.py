from app.ingestion import get_video_chunks
from app.vectorstore import get_or_create_vectorstore
from app.retriever import create_retriever
from app.chatbot import YouTubeChatbot
from dotenv import load_dotenv
from app.utils import extract_video_id

load_dotenv()

video_id = extract_video_id(
    "https://www.youtube.com/watch?v=oAkLSJNr5zY"
)

vector_store = get_or_create_vectorstore(
    video_id,
    lambda: get_video_chunks(video_id)
)

retriever = create_retriever(
    vector_store
)

bot = YouTubeChatbot(
    retriever,
    vector_store
)

for chunk in bot.stream_answer(
    "Summarize the video in 20 bullet points"
):
    print(chunk, end="", flush=True)