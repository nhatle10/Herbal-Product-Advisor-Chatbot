from typing import List, Tuple
from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np
from samples import chat_samples, tea_samples, test_samples
import argparse

class Route():
    def __init__(
        self,
        name: str = None,
        samples: List[str] = None
    ):
        self.name = name
        self.samples = samples or []

class SemanticRouter():
    def __init__(self, routes: List[Route], embedding: OpenAIEmbeddings):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {
            route.name: self.embedding.embed_documents(route.samples) for route in self.routes
        }

    def get_routes(self) -> List[Route]:
        return self.routes

    def guide(self, query: str) -> List[Tuple[float, str]]:
        queryEmbedding = self.embedding.embed_query(query)
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        scores = []

        for route in self.routes:
            route_embeddings = self.routesEmbedding[route.name]
            normalized_route_embeddings = route_embeddings / np.linalg.norm(route_embeddings, axis=1, keepdims=True)
            score = np.mean(np.dot(normalized_route_embeddings, queryEmbedding))
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores

def initialize_router(embedding_model_name: str, embedding_dimensions: int) -> SemanticRouter:
    """Initializes and returns a SemanticRouter instance."""
    embeddings = OpenAIEmbeddings(model=embedding_model_name, dimensions=embedding_dimensions)
    teaRoute = Route(name='tea', samples=tea_samples)
    chatRoute = Route(name='chat', samples=chat_samples)
    return SemanticRouter(routes=[teaRoute, chatRoute], embedding=embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize and run the Semantic Router.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Name of the embedding model to use.")
    parser.add_argument("--embedding_dimensions", type=int, default=1024, help="Number of dimensions for the embeddings.")

    args = parser.parse_args()

    router = initialize_router(args.embedding_model, args.embedding_dimensions)

    for sample in test_samples:
        route = router.guide(sample)
        print(route)