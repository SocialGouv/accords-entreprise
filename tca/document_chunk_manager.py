import time
from abc import ABC, abstractmethod
from typing import Optional, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from tca.custom_types import (
    Distance,
    DocumentID,
    DocumentName,
    Embeddings,
)
from tca.database.models import DocumentChunk, EmbeddingBase
from tca.embedding.embedding_clients import BaseEmbeddingClient
from tca.text.chunker import BaseChunker


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
        chunk_text: str,
        chunk_embedding: Embeddings,
        extra_metadata: dict,
    ) -> None:
        pass

    @abstractmethod
    def find_matching_documents(
        self, query_embedding: Embeddings, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        pass

    @abstractmethod
    def generate_embedding(self, text: list[str]) -> list[Embeddings]:
        pass

    @abstractmethod
    def chunk_document(self, document_text: str) -> list[str]:
        pass


class DummyDocumentChunkManager(BaseDocumentChunkManager):
    def __init__(self, chunker: Optional[BaseChunker]):
        super().__init__()
        self.chunker = chunker

    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: str,
        chunk_embedding: Embeddings,
        extra_metadata: dict,
    ) -> None:
        print("Dummy add_document_chunk called")

    def find_matching_documents(
        self, query_embedding: Embeddings, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        print("Dummy query_similar_chunks called")
        return []

    def generate_embedding(self, texts: list[str]) -> list[Embeddings]:
        # Placeholder method to generate embeddings for a given text
        # In a real scenario, this could call a machine learning model or an API
        return [[0.1, 0.2, 0.3] for _ in texts]  # Example embedding

    def chunk_document(self, document_text: str) -> list[str]:
        if not self.chunker:
            raise ValueError("Chunker is not provided")
        return self.chunker.build_chunks(document_text)


class DocumentChunkManager(BaseDocumentChunkManager):
    def __init__(
        self,
        session: Session,
        config: DocumentChunkManagerConfig,
        chunker: Optional[BaseChunker],
    ):
        super().__init__()
        self.session = session
        self.embedding_client = config["embedding_client"]
        self.db_embedding_model_cls = config["db_embedding_model_cls"]
        self.chunker = chunker

    def add_document_chunk(
        self,
        document_id: DocumentID,
        document_name: DocumentName,
        chunk_text: str,
        chunk_embeddings: Embeddings,
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
            embedding=chunk_embeddings,
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )
        self.session.add(embedding)
        self.session.commit()

    def find_matching_documents(
        self, query_embeddings: Embeddings, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        query = (
            select(
                DocumentChunk,
                self.db_embedding_model_cls.embedding.cosine_distance(
                    query_embeddings
                ).label("cosine_distance"),
            )
            .join(
                self.db_embedding_model_cls,
                self.db_embedding_model_cls.chunk_id == DocumentChunk.id,
            )
            .filter(
                self.db_embedding_model_cls.embedding.cosine_distance(query_embeddings)
                < cos_dist_threshold
            )
            .distinct(DocumentChunk.document_id)
            .order_by(
                DocumentChunk.document_id,
                "cosine_distance",
            )
        )

        results = self.session.execute(query).all()
        return [
            {
                "chunk": result.DocumentChunk,
                "distance": result.cosine_distance,
            }
            for result in results
        ]

    def generate_embedding(self, texts: list[str]) -> list[Embeddings]:
        return self.embedding_client.build_embedding(texts)

    def chunk_document(self, document_text: str) -> list[str]:
        if not self.chunker:
            raise ValueError("Chunker is not provided")
        return self.chunker.build_chunks(document_text)
