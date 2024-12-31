from typing import List
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import argparse
from modules.vector_store import load_vector_store
from langchain_openai.embeddings import OpenAIEmbeddings  # Import if needed
from sentence_transformers import SentenceTransformer # Import if needed

class Retriever:
    def __init__(self, vector_store: FAISS, bm25_retriever: BM25Retriever, reranker: FlagReranker, list_of_documents: List[Document], embeddings):
        self.semantic_retriever = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.list_of_documents = list_of_documents
        self.embeddings = embeddings 

    def __call__(self, query: str) -> List[Document]:
        semantic_results = self.semantic_retriever.similarity_search(
            query,
            k=10,
        )
        bm25_results = self.bm25_retriever.invoke(query)
        content = set()
        retrieval_docs = []

        for result in semantic_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)
        for result in bm25_results:
            if result.page_content not in content:
                content.add(result.page_content)
                retrieval_docs.append(result)

        pairs = [[query, doc.page_content] for doc in retrieval_docs]

        scores = self.reranker.compute_score(pairs, normalize=True)
        context_1 = []
        context_2 = []
        context = []
        parent_ids = set()
        for i in range(len(retrieval_docs)):
            if scores[i] >= 0.6:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_1.append(self.list_of_documents[parent_idx])
            elif scores[i] >= 0.1:
                parent_idx = retrieval_docs[i].metadata['parent']
                if parent_idx not in parent_ids:
                    parent_ids.add(parent_idx)
                    context_2.append(self.list_of_documents[parent_idx])

        if len(context_1) > 0:
            print('Context 1')
            context = context_1
        elif len(context_2) > 0:
            print('Context 2')
            context = context_2
        else:
            print('No relevant context')
        return context

def initialize_retriever(vector_store: FAISS, list_of_chunks: List[Document], list_of_documents: List[Document], reranker_model_name: str, embeddings) -> Retriever:
    """Initializes and returns a Retriever instance."""
    bm25_retriever = BM25Retriever.from_documents(
        list_of_chunks, k=10
    )
    reranker = FlagReranker(reranker_model_name, use_fp16=True)
    return Retriever(vector_store, bm25_retriever, reranker, list_of_documents, embeddings)

if __name__ == '__main__':
    from data_loader import load_documents_from_file
    from chunker import create_and_save_chunks, load_chunks_from_directory
    from langchain_openai.embeddings import OpenAIEmbeddings
    from sentence_transformers import SentenceTransformer
    from vector_store import load_vector_store
    
    parser = argparse.ArgumentParser(description="Initialize and run the Retriever.")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3", help="Name of the reranker model to use.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dimensions", type=int, default=1024, help="Number of dimensions for the embeddings.")
    parser.add_argument("--vector_store_path", type=str, default="data/vector_store", help="Path to the vector store directory.")
    parser.add_argument("--chunks_path", type=str, default="data/chunks_data", help="Path to the chunks directory.")
    parser.add_argument("--input_file", type=str, default="data/documents.txt", help="Path to the input document file.")
    parser.add_argument("--use_openai_embeddings", action="store_true", help="Use OpenAI embeddings. If not set, uses sentence transformers.")
    parser.add_argument("--sentence_transformer_model", type=str, default="hiieu/halong_embedding", help="Name of the sentence transformer model for local embeddings.")

    args = parser.parse_args()

    list_of_documents = load_documents_from_file(args.input_file)
    list_of_chunks = load_chunks_from_directory(args.chunks_path)
    if not list_of_chunks:
        list_of_chunks = create_and_save_chunks(
            list_of_documents,
            tokenizer_name=args.reranker_model,
            embedding_model_name=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            output_dir=args.chunks_path
        )

    if args.use_openai_embeddings:
        print("Using OpenAIEmbeddings for loading vector store...")
        embeddings = OpenAIEmbeddings(model=args.embedding_model, dimensions=args.embedding_dimensions)
    else:
        print(f"Using Sentence Transformer model ({args.sentence_transformer_model}) for loading vector store...")
        embeddings = SentenceTransformer(args.sentence_transformer_model)

    vector_store = load_vector_store(
        embeddings=embeddings,
        use_openai_embeddings=args.use_openai_embeddings,
        model_name=args.embedding_model if args.use_openai_embeddings else args.sentence_transformer_model,
        output_dir=args.vector_store_path,
    )

    retriever = initialize_retriever(vector_store, list_of_chunks, list_of_documents, args.reranker_model, embeddings)
    context = retriever("Tôi cần tư vấn về Trà hoa hồng")
    print(context)