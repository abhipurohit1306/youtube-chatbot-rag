from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

def get_youtube_prompt():
    return ChatPromptTemplate.from_messages([
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