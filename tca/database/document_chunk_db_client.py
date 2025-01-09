import time
from abc import ABC, abstractmethod
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from tca.custom_types import (
    Distance,
    DocumentID,
    DocumentName,
    Embeddings,
)
from tca.database.models import ChunkEmbeddingBase, DocumentChunk


class ResultChunkWithDistance(TypedDict):
    chunk: DocumentChunk
    distance: Distance


class BaseDocumentChunkDBClient(ABC):
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
        self, query_embeddings: Embeddings, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        pass


class DocumentChunkDBClient(BaseDocumentChunkDBClient):
    def __init__(
        self,
        session: Session,
        db_embedding_model_cls: type[ChunkEmbeddingBase],
    ):
        super().__init__()
        self.session = session
        self.db_embedding_model_cls = db_embedding_model_cls

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
