from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


video_id = "oAkLSJNr5zY" 
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=['en'])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    
    # print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)


# print(vector_store.index_to_docstore_id)



retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    llm=GoogleGenerativeAI(model="models/gemini-2.5-flash-lite")
)

result = retriever.invoke("What is deepmind? ")

# for i, doc in enumerate(result):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)



llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are an expert YouTube video assistant.

        Use the provided transcript context to answer the user's question.

        Instructions:
        - Base your answer primarily on the transcript context.
        - If the user asks for a summary, provide a concise but complete summary of the video.
        - If the user asks about a specific topic, extract and explain the relevant information from the transcript.
        - Combine information from multiple transcript sections when needed.
        - If the answer is partially available, provide the available information and mention what is missing.
        - Only say "I don't know" when the transcript contains no relevant information at all.
        - Keep the answer clear, structured, and easy to understand.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of cricket discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

def format_doc(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_doc),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke('give me the summary of the video?'))




# final_prompt = prompt.invoke({"context": context_text, "question": question})

# answer = llm.invoke(final_prompt)
# print(answer.content)




