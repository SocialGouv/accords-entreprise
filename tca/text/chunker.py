import re
from abc import ABC, abstractmethod

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tca.embedding.embedding_clients import BaseEmbeddingClient


class BaseChunker(ABC):
    @abstractmethod
    def build_chunks(self, document_text: str) -> list[str]:
        pass


class DelimiterChunker(BaseChunker):
    def __init__(self, delimiter_pattern: str = r"\.|\n\n?|\!|\?"):
        super().__init__()
        self.delimiter_pattern = delimiter_pattern

    def build_chunks(self, document_text: str) -> list[str]:
        chunks = re.split(self.delimiter_pattern, document_text)
        stripped_chunks = (chunk.strip() for chunk in chunks)
        non_empty_chunks = [paragraph for paragraph in stripped_chunks if paragraph]
        return non_empty_chunks


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_chunks(self, document_text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            keep_separator=True,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(document_text)


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        pre_chunker: BaseChunker,
        similarity_threshold: float = 0.8,
        min_chunk_size: int = 25,
        max_chunk_size: int = 150,
    ):
        super().__init__()
        self.pre_chunker = pre_chunker
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold

    def build_chunks(self, document_text: str) -> list[str]:
        simple_chunks = self.pre_chunker.build_chunks(document_text)
        chunks = []
        chunk_candidates = []
        current_embedding = None

        for simple_chunk in simple_chunks:
            chunk_candidates.append(simple_chunk)
            new_chunk_text = ". ".join(chunk_candidates)
            new_embedding = self.embedding_client.build_embedding([new_chunk_text])

            if current_embedding is None:
                current_embedding = new_embedding
                continue

            similarity = np.dot(current_embedding, np.array(new_embedding).T) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(new_embedding)
            )

            # If the similarity is below the threshold and the chunk size is not too small,
            # we consider the current doc_split as the start of a new chunk
            if (
                similarity < self.similarity_threshold
                and len(new_chunk_text) > self.min_chunk_size
            ) or len(new_chunk_text) > self.max_chunk_size:
                new_chunk = ". ".join(chunk_candidates[:-1])
                chunks.append(new_chunk)
                chunk_candidates = [simple_chunk]
                current_embedding = self.embedding_client.build_embedding(
                    [simple_chunk]
                )
            else:
                current_embedding = new_embedding

        if chunk_candidates:
            chunks.append(". ".join(chunk_candidates))

        return chunks
