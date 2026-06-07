from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from app.ingestion import get_video_chunks
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.prompts import get_youtube_prompt
from app.chatbot import YouTubeChatbot
from dotenv import load_dotenv
load_dotenv()

video_id = "oAkLSJNr5zY" 
chunks = get_video_chunks(video_id)


vector_store = get_or_create_vectorstore(video_id,chunks)

retriever = create_retriever(vector_store)

prompt = get_youtube_prompt()


bot = YouTubeChatbot(retriever)

bot.chat()