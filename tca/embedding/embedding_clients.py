from typing import Optional

import numpy as np
import ollama

# from FlagEmbedding import FlagLLMModel
from openai import OpenAI

from tca.custom_types import Embeddings


class BaseEmbeddingClient:
    def encode_corpus(self, documents: list[str]) -> list[Embeddings]:
        embeddings = self._encode_corpus(documents)
        # return self._normalize_embeddings(embeddings)
        return embeddings

    def encode_queries(self, queries: list[str]) -> list[Embeddings]:
        embeddings = self._encode_queries(queries)
        # return self._normalize_embeddings(embeddings)
        return embeddings

    def _normalize_embeddings(self, embeddings: list[Embeddings]) -> list[Embeddings]:
        normalized_embeddings = [
            np.array(embedding) / np.linalg.norm(embedding) for embedding in embeddings
        ]
        normalized_embeddings = [
            embedding.tolist() for embedding in normalized_embeddings
        ]
        return normalized_embeddings

    def _encode_corpus(self, documents: list[str]) -> list[Embeddings]:
        raise NotImplementedError("Subclasses should implement this method")

    def _encode_queries(self, queries: list[str]) -> list[Embeddings]:
        raise NotImplementedError("Subclasses should implement this method")


class OllamaEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model="bge-m3:567m-fp16"):
        self.model = model

    def _encode(self, texts: list[str]) -> list[Embeddings]:
        embeddings_response = ollama.embed(
            model=self.model,
            input=texts,
        )
        return embeddings_response["embeddings"]

    def _encode_corpus(self, documents: list[str]) -> list[Embeddings]:
        return self._encode(documents)

    def _encode_queries(self, queries: list[str]) -> list[Embeddings]:
        return self._encode(queries)


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        model="text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _encode(self, texts: list[str]) -> list[Embeddings]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def _encode_corpus(self, documents: list[str]) -> list[Embeddings]:
        return self._encode(documents)

    def _encode_queries(self, queries: list[str]) -> list[Embeddings]:
        return self._encode(queries)


# class FlagLLMEmbeddingClient(BaseEmbeddingClient):
#     def __init__(
#         self,
#         model_name="BAAI/bge-multilingual-gemma2",
#     ):
#         self.model = FlagLLMModel(
#             model_name_or_path=model_name,
#             # query_instruction_for_retrieval="Étant donné une requête de recherche sur le web, récupérer les passages pertinents qui répondent à la requête.",
#         )

#     def _encode_corpus(self, documents: list[str]) -> list[Embeddings]:
#         embeddings = self.model.encode_corpus(documents)
#         return embeddings.tolist()

#     def _encode_queries(self, queries: list[str]) -> list[Embeddings]:
#         embeddings = self.model.encode_queries(queries)
#         return embeddings.tolist()
