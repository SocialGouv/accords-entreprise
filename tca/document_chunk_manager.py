import time
from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from tca.custom_types import (
    ChunkText,
    DocumentID,
    DocumentName,
    DocumentText,
    Embedding,
)
from tca.models import DocumentChunk3D
from tca.vector_utils import VectorUtils


class ResultChunkWithSimilarity(TypedDict):
    chunk: DocumentChunk3D
    cosine_similarity: float


class BaseDocumentChunkManager(ABC):
    @abstractmethod
    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: ChunkText,
        chunk_embedding: Embedding,
        extra_metadata: dict,
    ) -> None:
        pass

    @abstractmethod
    def query_similar_chunks(
        self, query_embedding: Embedding, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithSimilarity]:
        pass

    @abstractmethod
    def generate_embedding(self, text: DocumentText) -> Embedding:
        pass

    @abstractmethod
    def chunk_document(self, document_text: DocumentText) -> list[ChunkText]:
        pass


class DummyDocumentChunkManager(BaseDocumentChunkManager):
    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: ChunkText,
        chunk_embedding: Embedding,
        extra_metadata: dict,
    ) -> None:
        print("Dummy add_document_chunk called")

    def query_similar_chunks(
        self, query_embedding: Embedding, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithSimilarity]:
        print("Dummy query_similar_chunks called")
        return []

    def generate_embedding(self, text):
        # Placeholder method to generate embeddings for a given text
        # In a real scenario, this could call a machine learning model or an API
        return [0.1, 0.2, 0.3]  # Example embedding

    def chunk_document(self, document_text: DocumentText) -> list[ChunkText]:
        # Example chunking logic: split by paragraphs
        return [chunk.strip() for chunk in document_text.split("\n\n")]


class DocumentChunkManager(BaseDocumentChunkManager):
    def __init__(self, session: Session):
        self.session = session

    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: ChunkText,
        chunk_embedding: Embedding,
        extra_metadata: dict,
    ) -> None:
        chunk = DocumentChunk3D(
            document_id=document_id,
            document_name=document_name,
            chunk_text=chunk_text,
            embedding=chunk_embedding,
            version=1,
            extra_metadata=extra_metadata or {},
            status="UP_TO_DATE",
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        self.session.add(chunk)
        self.session.commit()

    def query_similar_chunks(
        self, query_embedding: Embedding, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithSimilarity]:
        results = self.session.scalars(
            select(DocumentChunk3D)
            .filter(
                DocumentChunk3D.embedding.cosine_distance(query_embedding)
                < cos_dist_threshold
            )
            .order_by(DocumentChunk3D.embedding.cosine_distance(query_embedding))
        )

        chunks_with_similarity = []
        for chunk in results:
            cos_similarity = VectorUtils.cosine_similarity(
                np.array(query_embedding), np.array(chunk.embedding)
            )

            chunks_with_similarity.append(
                ResultChunkWithSimilarity(chunk=chunk, cosine_similarity=cos_similarity)
            )

        return chunks_with_similarity

    # TODO: Implement
    def generate_embedding(self, text):
        # Placeholder method to generate embeddings for a given text
        # In a real scenario, this could call a machine learning model or an API
        return [0.1, 0.2, 0.3]  # Example embedding

    # TODO: Implement
    def chunk_document(self, document_text: DocumentText) -> list[ChunkText]:
        # Example chunking logic: split by paragraphs
        return [chunk.strip() for chunk in document_text.split("\n\n")]
