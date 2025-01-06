from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tca.custom_types import (
    ChunkID,
    ChunkMetadata,
    ChunkStatus,
    DocumentID,
    DocumentName,
    Embeddings,
    MetadataVersion,
    TimestampSecond,
)


class PostgreSQLBase(DeclarativeBase):
    created_at: Mapped[TimestampSecond] = mapped_column(nullable=False)
    updated_at: Mapped[TimestampSecond] = mapped_column(nullable=False)


class DocumentChunk(PostgreSQLBase):
    __tablename__ = "document_chunks"

    id: Mapped[ChunkID] = mapped_column(primary_key=True)
    document_id: Mapped[DocumentID] = mapped_column(nullable=False)
    document_name: Mapped[DocumentName] = mapped_column(nullable=False)
    chunk_text: Mapped[str] = mapped_column(nullable=False)
    extra_metadata: Mapped[ChunkMetadata] = mapped_column(JSON)
    version: Mapped[MetadataVersion] = mapped_column(nullable=False, default=1)
    status: Mapped[ChunkStatus] = mapped_column(nullable=False, default="UP_TO_DATE")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, document_name={self.document_name}, document_id={self.document_id}, "
            f"extra_metadata={self.extra_metadata}, version={self.version}, "
            f"status={self.status})>\n"
            f'chunk_text:"{self.chunk_text}"'
        )


class EmbeddingBase(PostgreSQLBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[ChunkID] = mapped_column(
        ForeignKey("document_chunks.id"), nullable=False
    )
    embedding: Mapped[Embeddings]


class OllamaBgeM3Embedding(EmbeddingBase):
    __tablename__ = "ollama_bge_m3_embeddings"

    embedding: Mapped[Embeddings] = mapped_column(Vector(1024), nullable=False)
    __table_args__ = (
        Index(
            # See https://github.com/pgvector/pgvector/blob/master/README.md#hnsw
            "ix_ollama_bge_m3_embeddings_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "public.vector_cosine_ops"},
        ),
    )
