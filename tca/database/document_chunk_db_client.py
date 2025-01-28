import time
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tca.custom_types import (
    Distance,
    DocumentID,
    DocumentName,
    Embeddings,
)
from tca.database.models import (
    ChunkEmbeddingBase,
    DocumentChunk,
)


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
        raise NotImplementedError

    @abstractmethod
    def find_matching_documents_with_cos_dist(
        self, query_embeddings: Embeddings, cos_dist_threshold=0.5
    ) -> list[ResultChunkWithDistance]:
        raise NotImplementedError

    @abstractmethod
    def query_chunks_by_theme(
        self,
        theme_embeddings: Any,
        min_cos_dist_threshold: float = 0.75,
        nb_chunks_to_retrieve: int = 10,
    ) -> list[tuple[str, str, list[str], list[float]]]:
        raise NotImplementedError


class DocumentChunkDBClient(BaseDocumentChunkDBClient):
    def __init__(
        self,
        session: Session,
        chunk_embedding_cls: type[ChunkEmbeddingBase],
    ):
        super().__init__()
        self.session = session
        self.db_embedding_model_cls = chunk_embedding_cls

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
            embeddings=chunk_embeddings,
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )
        self.session.add(embedding)
        self.session.commit()

    def find_matching_documents_with_cos_dist(
        self, query_embeddings: Embeddings, cos_dist_threshold=0.3
    ) -> list[ResultChunkWithDistance]:
        query = (
            select(
                DocumentChunk,
                self.db_embedding_model_cls.embeddings.cosine_distance(
                    query_embeddings
                ).label("cosine_distance"),
            )
            .join(
                self.db_embedding_model_cls,
                self.db_embedding_model_cls.chunk_id == DocumentChunk.id,
            )
            .filter(
                self.db_embedding_model_cls.embeddings.cosine_distance(query_embeddings)
                < cos_dist_threshold
            )
            .order_by(
                DocumentChunk.document_id,
                "cosine_distance",
            )
            .distinct(DocumentChunk.document_id)
        )

        results = self.session.execute(query).all()
        return [
            {
                "chunk": result.DocumentChunk,
                "distance": result.cosine_distance,
            }
            for result in results
        ]

    def query_chunks_by_theme(
        self,
        theme_embeddings: Any,
        min_cos_dist_threshold: float = 0.75,
        nb_chunks_to_retrieve: int = 10,
    ) -> list[tuple[str, str, list[str], list[float]]]:
        """
        Query the database for chunks matching a given theme's embeddings.
        Returns a list of tuples containing document ID, document name, chunk texts, and cosine distances.
        """
        chunk_select_query = (
            select(
                DocumentChunk,
                self.db_embedding_model_cls.embeddings.cosine_distance(
                    theme_embeddings
                ).label("cos_distance"),
                func.row_number()
                .over(
                    partition_by=DocumentChunk.document_id,
                    order_by=self.db_embedding_model_cls.embeddings.cosine_distance(
                        theme_embeddings
                    ),
                )
                .label("row_number"),
            )
            .join(
                self.db_embedding_model_cls,
                self.db_embedding_model_cls.chunk_id == DocumentChunk.id,
            )
            .filter(
                self.db_embedding_model_cls.embeddings.cosine_distance(theme_embeddings)
                < min_cos_dist_threshold
            )
            .subquery()
        )

        query = (
            select(
                chunk_select_query.c.document_id,
                func.min(chunk_select_query.c.document_name).label("document_name"),
                func.array_agg(chunk_select_query.c.chunk_text).label("chunk_texts"),
                func.array_agg(chunk_select_query.c.cos_distance).label(
                    "cos_distances"
                ),
            )
            .filter(chunk_select_query.c.row_number <= nb_chunks_to_retrieve)
            .group_by(chunk_select_query.c.document_id)
            .order_by(func.min(chunk_select_query.c.cos_distance))
        )

        return self.session.execute(query).all()  # type: ignore
