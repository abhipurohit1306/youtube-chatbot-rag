from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from app.ingestion import get_video_chunks
from dotenv import load_dotenv
load_dotenv()



video_id = "oAkLSJNr5zY" 
# try:
#     ytt_api = YouTubeTranscriptApi()
#     transcript_list = ytt_api.fetch(video_id, languages=['en'])

#     transcript = " ".join(chunk.text for chunk in transcript_list)
    

# except TranscriptsDisabled:
#     print("Transcripts are disabled for this video.")


# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.create_documents([transcript])

chunks = get_video_chunks(video_id)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4}),
    llm=GoogleGenerativeAI(model="models/gemini-2.5-flash-lite")
)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert YouTube video assistant.

        Use the provided transcript context to answer the user's question.

        Instructions:
        - Base your answer primarily on the transcript context.
        - If the user asks for a summary, provide a concise but complete summary.
        - If the user asks about a topic, explain it clearly.
        - Combine information from multiple transcript sections when needed.
        - Only say "I don't know" if the transcript contains no relevant information.
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


def format_doc(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


chat_history = []


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_doc),
    "question": RunnablePassthrough(),
    "chat_history": RunnableLambda(
        lambda _: chat_history
    )
})

main_chain = parallel_chain | prompt | llm | parser

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    answer = main_chain.invoke(question)

    print("\nAI:")
    print(answer)

    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )




