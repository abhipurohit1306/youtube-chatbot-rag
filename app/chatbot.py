from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from app.prompts import get_youtube_prompt
from app.utils import format_timestamp
from app.utils import is_small_talk

class YouTubeChatbot:

    def __init__(self, retriever, vector_store):
        self.retriever = retriever
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-3.5-flash",
            temperature=0.2
        )
        self.prompt = get_youtube_prompt()
        self.chat_history = []
        self.parser = StrOutputParser()
        self.parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(self.format_doc),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(
                lambda _: self.chat_history
            )
        })
        self.main_chain = (self.parallel_chain | self.prompt | self.llm | self.parser)


    def format_doc(self,  retrieved_docs):
        return "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )
    
    def ask(self, question):

        if is_small_talk(question):
            answer = self.llm.invoke(question).content
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            return answer

        results = self.vector_store.similarity_search_with_score(
            question,
            k=4
        )

        best_score = results[0][1]
        print(f"Best Score: {best_score}")
        if best_score > 1.5:
            answer = (
                "I could not find sufficiently relevant information "
                "in the video to answer that question."
            )

            self.chat_history.append(
                HumanMessage(content=question)
            )

            self.chat_history.append(
                AIMessage(content=answer)
            )

            return answer

        retrieved_docs = [
            doc
            for doc, score in results
        ]

        answer = self.main_chain.invoke(
            question
        )

        citations = self.get_citations(
            retrieved_docs
        )
        negative_phrases = [
            "not discussed",
            "not mentioned",
            "does not discuss",
            "doesn't discuss",
            "no information",
            "not covered",
            "don't know"
        ]
        if any(
            phrase in answer.lower()
            for phrase in negative_phrases):
                return answer
        answer = (
            answer
            + "\n\nSources:\n"
            + "\n".join(citations)
        )

        self.chat_history.append(
            HumanMessage(content=question)
        )

        self.chat_history.append(
            AIMessage(content=answer)
        )
        # for doc, score in results:
        #     print(score)
        return answer


    def chat(self):
        while True:
            question = input("\nYou: ")
            if question.lower() == "exit":
                break
            answer = self.ask(
                question
            )
            print("\nAI:")
            print(answer)

    def get_citations(self, retrieved_docs):
        citations = []
        for doc in retrieved_docs:
            start = format_timestamp(doc.metadata["start"])
            end = format_timestamp(doc.metadata["end"])
            citations.append(f"[{start} - {end}]")

        return citations
            