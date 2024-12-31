import json
from typing import List
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import os
import argparse

def initialize_text_splitter(embedding_model_name: str, embedding_dimensions: int) -> SemanticChunker:
    """Initializes and returns the SemanticChunker."""
    embeddings = OpenAIEmbeddings(model=embedding_model_name, dimensions=embedding_dimensions)
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )

def create_and_save_chunks(
    list_of_documents: List[Document],
    tokenizer_name: str,
    embedding_model_name: str,
    embedding_dimensions: int,
    output_dir: str = "data/chunks_data",
) -> List[Document]:
    """Splits documents into chunks using SemanticChunker and saves them to a JSON file."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = initialize_text_splitter(embedding_model_name, embedding_dimensions)
    list_of_chunks = []
    for idx, doc in enumerate(list_of_documents):
        chunks = text_splitter.create_documents([doc.page_content])
        print(f'Document {idx}: Number of chunks: {len(chunks)} - Tokens of each chunk', end=' ')
        for chunk in chunks:
            text = chunk.page_content
            tokens = tokenizer.tokenize(text)
            num_tokens = len(tokens)
            if num_tokens > 1:
                chunk.metadata['parent'] = idx
                list_of_chunks.append(chunk)
            print(num_tokens, end=' ')
        print()

    print('Total chunks:', len(list_of_chunks))

    chunk_data = []
    for chunk in list_of_chunks:
        chunk_info = {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        chunk_data.append(chunk_info)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize the tokenizer name for the filename
    safe_tokenizer_name = tokenizer_name.replace("/", "--")
    output_file = os.path.join(output_dir, f"chunks_{safe_tokenizer_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    print(f"Chunks saved to {output_file}")
    return list_of_chunks

def load_chunks_from_file(input_file: str) -> List[Document]:
    """Loads chunks from a JSON file and returns them as a list of Documents."""
    with open(input_file, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    list_of_chunks = []
    for chunk_info in chunk_data:
        chunk = Document(
            page_content=chunk_info["content"],
            metadata=chunk_info["metadata"],
        )
        list_of_chunks.append(chunk)

    print(f"Loaded {len(list_of_chunks)} chunks from {input_file}")
    return list_of_chunks

def load_chunks_from_directory(input_dir: str = "data/chunks_data") -> List[Document]:
    """Loads all chunk files from a specified directory."""
    list_of_chunks = []
    for filename in os.listdir(input_dir):
        if filename.startswith("chunks_") and filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            list_of_chunks.extend(load_chunks_from_file(filepath))
    print(f"Loaded a total of {len(list_of_chunks)} chunks from directory '{input_dir}'")
    return list_of_chunks

if __name__ == '__main__':
    from data_loader import load_product_data, create_documents_from_data, load_documents_from_file

    parser = argparse.ArgumentParser(description="Chunk documents and save them to JSON files.")
    parser.add_argument("--input", type=str, default="data/documents.txt", help="Path to the input document file.")
    parser.add_argument("--output", type=str, default="data/chunks_data", help="Path to the output directory for chunks.")
    parser.add_argument("--mode", choices=["create", "load"], default="create", help="Whether to create new chunks or load from files.")
    parser.add_argument("--tokenizer", type=str, default="BAAI/bge-reranker-v2-m3", help="Name of the tokenizer to use.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dimensions", type=int, default=1024, help="Number of dimensions for the embeddings.")

    args = parser.parse_args()

    if args.mode == "create":
        list_of_documents = load_documents_from_file(args.input)
        list_of_chunks = create_and_save_chunks(
            list_of_documents,
            tokenizer_name=args.tokenizer,
            embedding_model_name=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            output_dir=args.output
        )
        print(f"Example chunk: {list_of_chunks[0].page_content[:100]}...")
    elif args.mode == "load":
        list_of_chunks = load_chunks_from_directory(args.output)
        if list_of_chunks:
            print(f"Example chunk: {list_of_chunks[0].page_content[:100]}...")
    else:
        print("Invalid mode specified. Use 'create' or 'load'.")