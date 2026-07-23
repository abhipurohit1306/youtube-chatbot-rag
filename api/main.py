from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import extract_video_id
from app.ingestion import get_video_chunks
from app.vectorstore import get_or_create_vectorstore
from app.retriever import create_retriever
from app.chatbot import YouTubeChatbot
from dotenv import load_dotenv
load_dotenv()


class VideoRequest(BaseModel):
    video_url: str

class ChatRequest(BaseModel):
    question: str

app = FastAPI()
chatbot = None



@app.get("/")
def home():

    return {
        "message": "YouTube Chatbot API"
    }

@app.post("/process-video")
def process_video(request: VideoRequest):

    global chatbot
    video_id = extract_video_id(request.video_url)
    if not video_id:
        return {
            "status": "error",
            "message": "Invalid YouTube URL"
        }
    vector_store = get_or_create_vectorstore(
        video_id,
        lambda: get_video_chunks(video_id)
    )
    retriever = create_retriever(
        vector_store
    )

    chatbot = YouTubeChatbot(retriever, vector_store)

    return {
        "status": "success"
    }



@app.post("/chat")
def chat(request: ChatRequest):

    global chatbot

    if chatbot is None:

        return {
            "status": "error",
            "message": "Process a video first"
        }

    answer = chatbot.ask(
        request.question
    )

    return {
        "answer": answer
    }

