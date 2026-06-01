from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from app.ingestion import get_video_chunks
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.prompts import get_youtube_prompt
from dotenv import load_dotenv
load_dotenv()



video_id = "oAkLSJNr5zY" 
chunks = get_video_chunks(video_id)



vector_store = create_vectorstore(chunks)

retriever = create_retriever(vector_store)

prompt = get_youtube_prompt()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)




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




