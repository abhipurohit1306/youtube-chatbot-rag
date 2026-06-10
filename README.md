# YouTube Chatbot using RAG

An AI-powered chatbot that enables users to interact with YouTube videos through natural language conversations. The application uses Retrieval-Augmented Generation (RAG) to extract video transcripts, retrieve relevant context, and generate accurate responses grounded in the video content.

## Features

* Extracts transcripts from YouTube videos
* Transcript chunking with LangChain
* Semantic search using FAISS vector database
* HuggingFace sentence-transformer embeddings
* Google Gemini-powered answer generation
* Conversational memory for follow-up questions
* Timestamp-based source citations
* Persistent FAISS index caching
* Similarity score filtering to reduce hallucinations
* Small-talk detection for natural conversations
* Streamlit frontend
* FastAPI backend with REST endpoints

## Architecture

```text
User
  ↓
Streamlit Frontend
  ↓ HTTP
FastAPI Backend
  ↓
YouTube Transcript API
  ↓
Chunking (LangChain)
  ↓
Embeddings (all-MiniLM-L6-v2)
  ↓
FAISS Vector Store
  ↓
Retriever
  ↓
Gemini 2.5 Flash Lite
  ↓
Response + Timestamp Citations
```

## Tech Stack

### Backend

* Python
* FastAPI
* LangChain

### LLM & Embeddings

* Google Gemini 2.5 Flash Lite
* HuggingFace Embeddings
* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* FAISS

### Frontend

* Streamlit

### Data Sources

* YouTube Transcript API

## Installation

```bash
git clone <repository-url>
cd YoutubeChatbot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

## API Endpoints

### Process Video

```http
POST /process-video
```

Request:

```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "question": "What is AsyncIO?"
}
```

## Project Structure

```text
YoutubeChatbot/
│
├── api/
│   └── main.py
│
├── app/
│   ├── chatbot.py
│   ├── ingestion.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── utils.py
│   └── vectorstore.py
│
├── faiss_indexes/
├── streamlit_app.py
├── yt_chatbot.py
├── requirements.txt
└── README.md
```

## Key RAG Enhancements

* Persistent FAISS storage to avoid rebuilding embeddings for previously processed videos
* Timestamp-aware retrieval for source attribution
* Similarity-score filtering to prevent irrelevant retrievals
* Citation suppression for unsupported answers
* Conversational memory using chat history

## Future Improvements

* Multi-video knowledge base
* Hybrid retrieval (BM25 + Vector Search)
* Response streaming
* Docker deployment
* User authentication
* Cloud deployment (Render/Railway + Streamlit Cloud)

## Sample Questions

* Summarize the video
* What is AsyncIO?
* Explain event loops
* What are the key takeaways?
* Show where this topic was discussed

## Author
Abhishek Purohit
