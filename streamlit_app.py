import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title('Youtube Chatbot')

video_url = st.text_input("Enter YouTube URL",key="video_url")

if st.button("Process Video"):
    st.session_state.messages = []

    response = requests.post(
        f"{API_URL}/process-video",
        json={
            "video_url": video_url
        }
    )

    if response.status_code != 200:
        st.error(
            f"Backend Error ({response.status_code})"
        )
        st.write(response.text)
        st.stop()

    data = response.json()

    if data["status"] == "success":
        st.success("Video Ready")
    else:
        st.error(data["message"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])  

question = st.chat_input("Ask about the video")
if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "question": question
        }
    )

    if response.status_code != 200:
        st.error(
            f"Backend Error ({response.status_code})"
        )
        st.write(response.text)
        st.stop()

    data = response.json()

    if "answer" in data:

        answer = data["answer"]

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:
        st.warning(data["message"])

st.sidebar.title("YouTube Chatbot")
st.sidebar.markdown("---")

if st.sidebar.button("🎥 New Video"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()