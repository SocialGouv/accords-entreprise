import ollama

from tca.custom_types import Embedding


class BaseEmbeddingClient:
    def embed(self, texts: list[str]) -> list[Embedding]:
        raise NotImplementedError("Subclasses should implement this method")


class OllamaEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model="bge-m3:567m-fp16"):
        self.model = model

    def embed(self, texts: list[str]) -> list[Embedding]:
        embeddings_response = ollama.embed(
            model=self.model,
            input=texts,
        )
        return embeddings_response["embeddings"]
