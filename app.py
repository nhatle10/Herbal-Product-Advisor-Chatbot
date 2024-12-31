import os
from typing import List, Dict, Callable, Any
from pyngrok import ngrok, conf
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import re
import logging

from modules.data_loader import load_product_data, load_documents_from_file
from modules.chunker import load_chunks_from_file
from modules.vector_store import initialize_embedding, load_vector_store
from modules.retriever import initialize_retriever  # Ensure correct import
from modules.reflection import initialize_reflection, Reflection  # Import Reflection class
from modules.llm_generator import initialize_llm_chain
from modules.tools import ContactShop, RefuseToAnswer, Retrieve
from config.config import (MODEL_NAME, TEMPERATURE,
                            USE_OPENAI_EMBEDDINGS, OPENAI_EMBEDDING_MODEL,
                            OPENAI_EMBEDDING_DIMENSIONS, SENTENCE_TRANSFORMER_MODEL,
                            RERANKER_MODEL)

# Initialize components
list_of_documents = load_documents_from_file("data/documents.txt")
list_of_chunks = load_chunks_from_file("data/chunks_data/chunks_BAAI--bge-reranker-v2-m3.json")

embeddings = initialize_embedding(
    USE_OPENAI_EMBEDDINGS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSIONS,
    SENTENCE_TRANSFORMER_MODEL
)

vector_store = load_vector_store(embeddings, use_openai_embeddings=USE_OPENAI_EMBEDDINGS, model_name=OPENAI_EMBEDDING_MODEL if USE_OPENAI_EMBEDDINGS else SENTENCE_TRANSFORMER_MODEL, output_dir='data/vector_store')
retriever = initialize_retriever(vector_store, list_of_chunks, list_of_documents, RERANKER_MODEL, embeddings) # Pass embeddings here

reflection: Reflection = initialize_reflection(MODEL_NAME, TEMPERATURE) # Initialize reflection here
agentChain, answerPrompt, answerModel = initialize_llm_chain(MODEL_NAME, TEMPERATURE)
answerChain = answerPrompt | answerModel

app = Flask(__name__)
CORS(app)

def stream_response(generator: Callable[[], Any]) -> Response:
    """Streams a response from a generator."""
    return Response(generator(), mimetype='text/plain')

@app.route('/v2/chat', methods=['POST'])
def chat_v2() -> Response:
    """
    Chat endpoint leveraging function calling (tools).
    """
    user_message: Dict = request.json.get('message', {})
    context: List[Dict] = request.json.get('context', [])
    stream: bool = request.json.get('stream', True)

    print(f'[/v2/chat] Message: {user_message}')
    print(f'[/v2/chat] Context: {context}')

    history_string = "\n".join(f"{message['role']} : {message['content']}" for message in context)
    print(f"[/v2/chat] History String: {history_string}")

    # Use reflection to refine the query
    try:
        refined_query = reflection(context)
        print(f"[/v2/chat] Refined Query: {refined_query}")
    except Exception as e:
        logging.error(f"[/v2/chat] Error in reflection: {e}")
        refined_query = user_message['content'] # Fallback to original message
        print(f"[/v2/chat] Using original query due to error: {refined_query}")

    try:
        agent_response = agentChain.invoke({"chat_history": history_string})
    except Exception as e:
        logging.error(f"[/v2/chat] Error invoking agentChain: {e}")
        return jsonify({'error': f"Error invoking agentChain: {e}"})

    if agent_response.additional_kwargs.get('tool_calls'):
        tool_calls_data = agent_response.additional_kwargs['tool_calls']
        for tool_call_data in tool_calls_data:
            function_call = tool_call_data['function']
            tool_name = function_call['name']
            arguments = json.loads(function_call['arguments'])

            if tool_name == 'Retrieve':
                try:
                    print(f'[/v2/chat] Retrieve Tool - Query: {refined_query}')
                    retrieved_context = retriever(refined_query)
                    print(f"[/v2/chat] Retrieved Context: {retrieved_context}")
                    source_information = "\n".join(
                        f"{doc.page_content} - Link ảnh: {doc.metadata.get('image_urls', 'N/A') if isinstance(doc.metadata, dict) else 'N/A'}"
                        for doc in retrieved_context
                    )
                    print(f"[/v2/chat] Source Information: {source_information}")
                    if stream:
                        def generate():
                            try:
                                for chunk in answerChain.stream({"query": refined_query, "source_information": source_information}):
                                    yield chunk.content
                            except Exception as e:
                                logging.error(f"[/v2/chat] Error in answerChain.stream: {e}")
                                yield f"Error: {e}"
                        return app.response_class(generate(), mimetype='text/plain')
                    else:
                        response = answerChain.invoke({"query": refined_query, "source_information": source_information})
                        return jsonify({'response': response.content})
                except Exception as e:
                    logging.error(f"[/v2/chat] Error in Retrieve tool: {e}")
                    return jsonify({'error': f"Error in Retrieve tool: {e}"})

            elif tool_name == 'ContactShop':
                print(f"[/v2/chat] ContactShop Tool Called")
                response_message = f"Nếu bạn có nhu cầu mua hàng xin vui lòng liên hệ cửa hàng qua số điện thoại 0902436989 hoặc email thaoduoctoanthang@gmail.com."
                if stream:
                    return app.response_class((chunk + " " for chunk in response_message.split()), mimetype='text/plain')
                else:
                    return jsonify({'response': response_message})

            elif tool_name == 'RefuseToAnswer':
                reason = arguments.get('reason', 'Tôi xin phép không trả lời câu hỏi này.')
                print(f"[/v2/chat] RefuseToAnswer Tool Called: {reason}")
                if stream:
                    return app.response_class((chunk + " " for chunk in reason.split()), mimetype='text/plain')
                else:
                    return jsonify({'response': reason})
    else:
        print('[/v2/chat] No tool call')
        if stream:
            def generate():
                try:
                    for chunk in agent_response.content.split(" "): # Consider direct stream if available
                        yield chunk + " "
                except Exception as e:
                    logging.error(f"[/v2/chat] Error streaming agent response: {e}")
                    yield f"Error: {e}"
            return app.response_class(generate(), mimetype='text/plain')
        else:
            return jsonify({'response': agent_response.content})

if __name__ == '__main__':
    # Start ngrok tunnel
    conf.get_default().region = "ap"
    public_url = ngrok.connect(8000)
    print(f" * ngrok tunnel: {public_url}")

    # Extract and save the ngrok URL
    match = re.search(r"\"(https?://.*?)\" ", str(public_url))
    if match:
        ngrok_url = match.group(1)
        print(f" * Extracted ngrok URL: {ngrok_url}")

        with open("./pages/ngrok_url.txt", "w") as f:
            f.write(ngrok_url)
    else:
        print(" * Error: Could not extract ngrok URL.")

    app.run(port=8000)
