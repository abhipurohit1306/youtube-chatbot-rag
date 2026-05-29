from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


video_id = "Gfr50f6ZBvo" 
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



retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

result = retriever.invoke("What is deepmind? ")

# for i, doc in enumerate(result):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)



llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

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

print(main_chain.invoke('What are the topics discussed in  the video?'))




# final_prompt = prompt.invoke({"context": context_text, "question": question})

# answer = llm.invoke(final_prompt)
# print(answer.content)




