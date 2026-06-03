import streamlit as st
from app.ingestion import get_video_chunks
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.chatbot import YouTubeChatbot
from dotenv import load_dotenv
load_dotenv()

if "bot" not in st.session_state:
    st.session_state.bot = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title('Youtube Chatbot')

video_id = st.text_input('Enter youtube video id') 

if st.button("Process Video"):
    st.session_state.messages = []

    chunks = get_video_chunks(video_id)

    vector_store = create_vectorstore(chunks)

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

