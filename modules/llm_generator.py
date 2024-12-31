from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
import argparse
import json
from .tools import Retrieve, RefuseToAnswer, ContactShop  
from .prompts import toolPrompt, answerPrompt 
from langchain_core.prompts import PromptTemplate

def initialize_llm_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.5) -> Tuple[BaseChatModel, PromptTemplate, BaseChatModel]:
    """Initializes and returns the LLM, agent chain, and answer model."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [Retrieve, RefuseToAnswer, ContactShop]
    agent = llm.bind_tools(tools)
    agentChain = toolPrompt | agent
    answerModel = ChatOpenAI(model=model_name, temperature=temperature)
    return agentChain, answerPrompt, answerModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize and run the LLM chain.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the LLM model to use.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for the LLM model.")

    args = parser.parse_args()

    agentChain, answerPrompt, answerModel = initialize_llm_chain(args.model_name, args.temperature)

    teaChatHistory = [{"role": "user", "content": "Xin chào shop"},
                   {"role": "assistant", "content": "Chào bạn tôi có thể giúp gì cho bạn"},
                   {"role": "user", "content": "Tôi cần tư vấn một số trà hoa cúc"},
                   {"role": "assistant", "content":"Bạn muốn tư vấn trà hoa cúc với hoa màu gì?"},
                   {"role": "user", "content": "Trà hoa cúc màu vàng"}]
    tea_history_string = "\n".join([f"{message['role']} : {message['content']}" for message in teaChatHistory])

    response = agentChain.invoke({"chat_history": tea_history_string + "\nUser: Tôi muốn mua trà hoa cúc"})
    print(f"Agent Response: {response}")
    if response.additional_kwargs.get('function_call'):
        function_call = response.additional_kwargs['function_call']
        tool_name = function_call['name']
        tool_arguments = json.loads(function_call['arguments'])
        print(f"Tool Call: {tool_name} with arguments: {tool_arguments}")

    chatHistory = [{"role": "user", "content": "Xin chào shop"},
                   {"role": "assistant", "content": "Chào bạn tôi có thể giúp gì cho bạn"},
                   {"role": "user", "content": "Quốc kỳ Việt Nam có bao nhiêu màu?"}]
    history_string = "\n".join([f"{message['role']} : {message['content']}" for message in chatHistory])

    normal_response = agentChain.invoke({"chat_history": history_string})
    print(f"Normal Response: {normal_response}")

    query = "Trà hoa cúc có tác dụng gì?"
    source_information = "Trà hoa cúc giúp thanh nhiệt, giải độc và an thần."
    answer_prompt_input = answerPrompt.invoke({"query": query, "source_information": source_information})
    final_answer = answerModel.invoke(answer_prompt_input)

    print(f"Final Answer: {final_answer.content}")