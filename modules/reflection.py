from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Optional, Type, List, Dict
import argparse

class Reflection:
    def __init__(self, llm):
        """
        Initialize the Reflection class with a language model (llm).
        """
        self.llm = llm

    def __call__(self, chat_history, last_items_considered=100):
        """
        Reformulates the latest user question in Vietnamese, making it standalone and understandable
        without the context of the chat history.

        Args:
            chat_history (list): A list of dictionaries representing the chat messages.
            last_items_considered (int): Number of latest messages to consider from the chat history.

        Returns:
            str: The reformulated standalone question in Vietnamese.
        """
        # Limit the chat history to the last `last_items_considered` messages if needed.
        if len(chat_history) > last_items_considered:
            chat_history = chat_history[-last_items_considered:]

        # Create a formatted string of the chat history excluding the latest message.
        history_string = "\n".join(
            f"{message['role']}: {message['content']}" for message in chat_history[:-1]
        )

        # Prepare the chat template prompt.
        chat_template = PromptTemplate.from_template(
            """Given a chat history and the latest user question which might reference context in the chat history,
            formulate a standalone question in Vietnamese which can be understood without the chat history.
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

            Chat History:
            {history_string}

            Latest User Question:
            {input}"""
        )

        # Chain the template with the language model to generate the response.
        chain = chat_template | self.llm
        response = chain.invoke({
            "input": chat_history[-1]["content"], 
            "history_string": history_string
        })

        # Return the content of the response.
        return response.content

def initialize_reflection(model_name: str, temperature: float) -> Reflection:
    """Initializes and returns a Reflection instance."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return Reflection(llm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize and run the Reflection process.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the LLM model to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the LLM model.")

    args = parser.parse_args()

    reflection = initialize_reflection(args.model_name, args.temperature)
    chatHistory = [{"role": "user", "content": "Xin chào shop"},
                   {"role": "assistant", "content": "Chào bạn tôi có thể giúp gì cho bạn"},
                   {"role": "user", "content": "Tôi cần tư vấn một loại trà thảo mộc"},
                   {"role": "assistant", "content": "Bạn muốn được tư vấn loại trà thảo mộc nào?"},
                   {"role": "user", "content": "Trà thảo mộc hoa cúc"}]

    refined_query = reflection(chatHistory)
    print(f"Refined query: {refined_query}")