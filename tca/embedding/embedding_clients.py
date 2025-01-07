import numpy as np
import ollama

from tca.custom_types import Embeddings


class BaseEmbeddingClient:
    def embed(self, texts: list[str]) -> list[Embeddings]:
        embeddings = self._embed(texts)
        normalized_embeddings = [
            np.array(embedding) / np.linalg.norm(embedding) for embedding in embeddings
        ]
        normalized_embeddings = [
            embedding.tolist() for embedding in normalized_embeddings
        ]
        return normalized_embeddings  # type: ignore

    def _embed(self, texts: list[str]) -> list[Embeddings]:
        raise NotImplementedError("Subclasses should implement this method")


class OllamaEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model="bge-m3:567m-fp16"):
        self.model = model

    def _embed(self, texts: list[str]) -> list[Embeddings]:
        embeddings_response = ollama.embed(
            model=self.model,
            input=texts,
        )
        return embeddings_response["embeddings"]
