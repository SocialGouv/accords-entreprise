import time
from abc import ABC, abstractmethod
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from tca.custom_types import (
    ChunkText,
    Distance,
    DocumentID,
    DocumentName,
    DocumentText,
    Embedding,
)
from tca.database.models import DocumentChunk, EmbeddingBase
from tca.embedding.embedding_clients import BaseEmbeddingClient


class DocumentChunkManagerConfig(TypedDict):
    embedding_client: BaseEmbeddingClient
    db_embedding_model_cls: type[EmbeddingBase]


class ResultChunkWithDistance(TypedDict):
    chunk: DocumentChunk
    distance: Distance


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
    ) -> list[ResultChunkWithDistance]:
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
    ) -> list[ResultChunkWithDistance]:
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
    def __init__(
        self,
        session: Session,
        config: DocumentChunkManagerConfig,
    ):
        self.session = session
        self.embedding_client = config["embedding_client"]
        self.db_embedding_model_cls = config["db_embedding_model_cls"]

    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: ChunkText,
        chunk_embedding: Embedding,
        extra_metadata: dict,
    ) -> None:
        current_timestamp = int(time.time())
        chunk = DocumentChunk(
            document_id=document_id,
            document_name=document_name,
            chunk_text=chunk_text,
            version=1,
            extra_metadata=extra_metadata or {},
            status="UP_TO_DATE",
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )
        self.session.add(chunk)
        self.session.flush()  # Ensure the chunk ID is generated
        embedding = self.db_embedding_model_cls(
            chunk_id=chunk.id,
            embedding=chunk_embedding,
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )
        self.session.add(embedding)
        self.session.commit()

    def query_similar_chunks(
        self, query_embeddings: Embedding, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        # Subquery to calculate cosine distance and filter by threshold
        embedding_subquery = (
            select(
                self.db_embedding_model_cls.chunk_id,
                self.db_embedding_model_cls.embedding.cosine_distance(
                    query_embeddings
                ).label("cosine_distance"),
            )
            .filter(
                self.db_embedding_model_cls.embedding.cosine_distance(query_embeddings)
                < cos_dist_threshold
            )
            .subquery()
        )

        # Join the subquery with DocumentChunk to get associated chunks
        query = (
            select(
                DocumentChunk,
                self.db_embedding_model_cls.embedding,
                embedding_subquery.c.cosine_distance,
            )
            .join(embedding_subquery, embedding_subquery.c.chunk_id == DocumentChunk.id)
            .filter(self.db_embedding_model_cls.chunk_id == DocumentChunk.id)
            .order_by(embedding_subquery.c.cosine_distance)
        )

        results = self.session.execute(query).all()
        return [
            {
                "chunk": result.DocumentChunk,
                "distance": result.cosine_distance,
            }
            for result in results
        ]

    def generate_embedding(self, text):
        # Placeholder method to generate embeddings for a given text
        # In a real scenario, this could call a machine learning model or an API
        # TODO: Maybe transform this parameter to a list of texts
        return self.embedding_client.embed([text])[0]

    def chunk_document(self, document_text: DocumentText) -> list[ChunkText]:
        # Example chunking logic: split by paragraphs
        return [chunk.strip() for chunk in document_text.split("\n\n")]
