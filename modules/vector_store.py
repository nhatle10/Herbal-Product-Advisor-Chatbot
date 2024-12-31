import json
from typing import List, Callable
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
import os
import argparse
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def initialize_embedding(
    use_openai_embeddings: bool,
    openai_embedding_model: str,
    openai_embedding_dimensions: int,
    sentence_transformer_model: str,
):
    if use_openai_embeddings:
        print(f"Using OpenAIEmbeddings model: {openai_embedding_model}")
        embeddings = OpenAIEmbeddings(model=openai_embedding_model, dimensions=openai_embedding_dimensions)

    else:
        print(f"Using Sentence Transformer model: {sentence_transformer_model}")
        embeddings = SentenceTransformer(sentence_transformer_model)

    return embeddings

def create_vector_store(
    list_of_chunks: List[Document],
    embeddings,
    model_name: str,
    output_dir: str = "data/vector_store",
    use_openai_embeddings: bool = False,
) -> FAISS:
    """Creates a FAISS vector store from a list of document chunks and saves it to files."""
    os.makedirs(output_dir, exist_ok=True)
    save_model_name = model_name.replace("/", "--").replace(" ", "_")
    index_file = os.path.join(output_dir, f"faiss_index_{save_model_name}.bin")
    metadata_file = os.path.join(output_dir, f"metadata_{save_model_name}.pkl")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))) if use_openai_embeddings else faiss.IndexFlatL2(embeddings.get_sentence_embedding_dimension())

    def get_embeddings(texts):
        return embeddings.encode(texts)
    
    vector_store = FAISS(
        embedding_function=embeddings if use_openai_embeddings else get_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(list_of_chunks))]
    print(uuids)
    vector_store.add_documents(documents=list_of_chunks, ids=uuids)

    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump({
            "docstore": vector_store.docstore,
            "index_to_docstore_id": vector_store.index_to_docstore_id,
        }, f)
    print(f"FAISS index saved to {index_file} and metadata to {metadata_file}")
    return vector_store

def load_vector_store(
    embeddings,
    model_name: str,
    output_dir: str = "data/vector_store",
    use_openai_embeddings: bool = False,
) -> FAISS:
    """Loads a FAISS vector store from files."""
    save_model_name = model_name.replace("/", "--").replace(" ", "_")
    index_file = os.path.join(output_dir, f"faiss_index_{save_model_name}.bin")
    metadata_file = os.path.join(output_dir, f"metadata_{save_model_name}.pkl")

    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    docstore = metadata["docstore"]
    index_to_docstore_id = metadata["index_to_docstore_id"]

    def get_embeddings(texts):
        return embeddings.encode(texts)
    vector_store = FAISS(
        embedding_function=embeddings if use_openai_embeddings else get_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    print(f"FAISS index loaded from {index_file} and metadata from {metadata_file}")
    return vector_store

if __name__ == '__main__':
    from data_loader import load_documents_from_file
    from chunker import create_and_save_chunks, load_chunks_from_directory

    parser = argparse.ArgumentParser(description="Chunk documents and create/load a vector store.")
    parser.add_argument("--input", type=str, default="data/documents.txt", help="Path to the input document file.")
    parser.add_argument("--chunks_output", type=str, default="data/chunks_data", help="Path to the output directory for chunks.")
    parser.add_argument("--vectorstore_output", type=str, default="data/vector_store", help="Path to the output directory for the vector store.")
    parser.add_argument("--chunk_mode", choices=["create", "load"], default="create", help="Whether to create new chunks or load from files.")
    parser.add_argument("--vectorstore_mode", choices=["create", "load"], default="create", help="Whether to create a new vector store or load from files.")
    parser.add_argument("--tokenizer", type=str, default="BAAI/bge-reranker-v2-m3", help="Name of the tokenizer to use.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dimensions", type=int, default=1024, help="Number of dimensions for the embeddings.")
    parser.add_argument("--sentence_transformer_model", type=str, default="hiieu/halong_embedding", help="Name of the sentence transformer model for local embeddings.")
    parser.add_argument("--use_openai_embeddings", action="store_true", help="Use OpenAI embeddings. If not set, uses sentence transformers.")

    args = parser.parse_args()

    list_of_chunks = []

    # Handle chunk creation or loading
    if args.chunk_mode == "create":
        list_of_documents = load_documents_from_file(args.input)
        list_of_chunks = create_and_save_chunks(
            list_of_documents,
            tokenizer_name=args.tokenizer,
            embedding_model_name=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            output_dir=args.chunks_output
        )
        if list_of_chunks:
            print(f"Example chunk: {list_of_chunks[0].page_content[:100]}...")
    elif args.chunk_mode == "load":
        list_of_chunks = load_chunks_from_directory(args.chunks_output)
        if list_of_chunks:
            print(f"Example chunk: {list_of_chunks[0].page_content[:100]}...")
    else:
        print("Invalid chunk mode specified. Use 'create' or 'load'.")

    embeddings = initialize_embedding(
        args.use_openai_embeddings,
        args.embedding_model,
        args.embedding_dimensions,
        args.sentence_transformer_model
    )

    if args.vectorstore_mode == "create":
        if list_of_chunks:
            print("Creating vector store...")
            create_vector_store(
                list_of_chunks=list_of_chunks,
                embeddings=embeddings,
                model_name=args.embedding_model if args.use_openai_embeddings else args.sentence_transformer_model,
                use_openai_embeddings=args.use_openai_embeddings,
                output_dir=args.vectorstore_output,
            )
        else:
            print("No chunks available to create a vector store. Please create or load chunks first.")
    elif args.vectorstore_mode == "load":
        print("Loading vector store...")
        load_vector_store(
            embeddings,
            args.embedding_model if args.use_openai_embeddings else args.sentence_transformer_model,
            args.vectorstore_output,
            args.use_openai_embeddings
        )
    else:
        print("Invalid vector store mode specified. Use 'create' or 'load'.")