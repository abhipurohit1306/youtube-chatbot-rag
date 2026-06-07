import streamlit as st
from app.ingestion import get_video_chunks
from app.vectorstore import get_or_create_vectorstore
from app.retriever import create_retriever
from app.chatbot import YouTubeChatbot
from dotenv import load_dotenv
from app.utils import extract_video_id
load_dotenv()

if "bot" not in st.session_state:
    st.session_state.bot = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title('Youtube Chatbot')

video_url = st.text_input("Enter YouTube URL",key="video_url")

if st.button("Process Video"):
    st.session_state.messages = []
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
    else:
        chunks = get_video_chunks(video_id)

    chunks = get_video_chunks(video_id)
    if not chunks:
        st.error("No transcript chunks found for this video.")
        st.stop()

    vector_store = get_or_create_vectorstore(video_id,chunks)

    retriever = create_retriever(vector_store)

    bot = YouTubeChatbot(retriever)

    st.session_state.bot = bot

    st.success("Video Ready")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])  

question = st.chat_input(
    "Ask about the video"
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    if st.session_state.bot is None:
        st.warning("Please process a video first.")
    else:
        answer = st.session_state.bot.ask(question)
        st.write(answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

st.sidebar.title("YouTube Chatbot")

st.sidebar.markdown("---")

if st.sidebar.button("🎥 New Video"):
    st.session_state.bot = None
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()