from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tca.custom_types import (
    ChunkID,
    ChunkMetadata,
    ChunkStatus,
    ChunkText,
    DocumentID,
    DocumentName,
    Embedding,
    MetadataVersion,
    TimestampSecond,
)


class PostgreSQLBase(DeclarativeBase):
    # see https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html#customizing-the-type-map
    type_annotation_map = {Embedding: Vector(3)}
    created_at: Mapped[TimestampSecond] = mapped_column(nullable=False)
    updated_at: Mapped[TimestampSecond] = mapped_column(nullable=False)


class DocumentChunk(PostgreSQLBase):
    __tablename__ = "document_chunks"

    id: Mapped[ChunkID] = mapped_column(primary_key=True)
    document_id: Mapped[DocumentID] = mapped_column(nullable=False)
    document_name: Mapped[DocumentName] = mapped_column(nullable=False)
    chunk_text: Mapped[ChunkText] = mapped_column(nullable=False)
    embedding: Mapped[Embedding] = mapped_column(nullable=False)
    extra_metadata: Mapped[ChunkMetadata] = mapped_column(JSON)
    # Adding a version column to handle schema changes in extra_metadata
    version: Mapped[MetadataVersion] = mapped_column(nullable=False, default=1)
    # Adding a status column to track the state of the document chunk
    status: Mapped[ChunkStatus] = mapped_column(nullable=False, default="UP_TO_DATE")

    __table_args__ = (Index("ix_document_chunks_embedding", "embedding"),)

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, document_name={self.document_name}, document_id={self.document_id}, "
            f"extra_metadata={self.extra_metadata}, version={self.version}, "
            f"status={self.status})>\n"
            f'chunk_text:"{self.chunk_text}"'
        )
