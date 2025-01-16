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
    ThemeID,
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


class ChunkEmbeddingBase(PostgreSQLBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[ChunkID] = mapped_column(
        ForeignKey("document_chunks.id"), nullable=False
    )
    embeddings: Mapped[Embeddings]


class BgeM3ChunkEmbedding(ChunkEmbeddingBase):
    __tablename__ = "bge_m3_chunk_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(1024), nullable=False)
    __table_args__ = (
        Index(
            # See https://github.com/pgvector/pgvector/blob/master/README.md#hnsw
            "ix_bge_m3_chunk_embeddings_embedding",
            "embeddings",
            postgresql_using="hnsw",
            postgresql_ops={"embeddings": "public.vector_cosine_ops"},
        ),
    )


class BGEMultilingualGemma2ChunkEmbedding(ChunkEmbeddingBase):
    __tablename__ = "bge_multilingual_gemma2_chunk_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(3584), nullable=False)
    # TODO: HNSW index does not support vectors of size 2000+ so we need to implement another index like
    # https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat
    # However it is more work as the index needs to be created after the documents are ingested
    # We can also try to make halfvecs work : https://github.com/pgvector/pgvector/issues/461#issuecomment-2365655349

    # __table_args__ = (
    #     Index(
    #         # See https://github.com/pgvector/pgvector/blob/master/README.md#hnsw
    #         "ix_bge_multilingual_gemma2_chunk_embeddings_embedding",
    #         "embeddings::halfvec(3584)",
    #         postgresql_using="hnsw",
    #         postgresql_ops={"embedding": "halfvec_cosine_ops"},
    #     ),
    # )


class OpenAITextEmbedding3LargeChunkEmbedding(ChunkEmbeddingBase):
    __tablename__ = "openai_text_embedding_3_large_chunk_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(3072), nullable=False)
    # TODO: Same as BGE above


class Theme(PostgreSQLBase):
    __tablename__ = "themes"

    id: Mapped[ThemeID] = mapped_column(primary_key=True)
    prompt: Mapped[str] = mapped_column(nullable=False, default="")
    description: Mapped[str] = mapped_column(nullable=False, default="")
    themes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    version: Mapped[MetadataVersion] = mapped_column(nullable=False, default=1)
    status: Mapped[ChunkStatus] = mapped_column(nullable=False, default="UP_TO_DATE")

    def __repr__(self) -> str:
        return (
            f"<ThemePrompt(id={self.id}, prompt={self.prompt}, description={self.description}, themes={self.themes}, "
            f"version={self.version}, status={self.status})>"
        )


class BaseThemeEmbedding(PostgreSQLBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    theme_id: Mapped[ThemeID] = mapped_column(ForeignKey("themes.id"), nullable=False)
    embeddings: Mapped[Embeddings]


class BgeM3ThemeEmbedding(BaseThemeEmbedding):
    __tablename__ = "bge_m3_theme_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(1024), nullable=False)
    __table_args__ = (
        Index(
            "ix_bge_m3_theme_embeddings_embedding",
            "embeddings",
            postgresql_using="hnsw",
            postgresql_ops={"embeddings": "public.vector_cosine_ops"},
        ),
    )


class BGEMultilingualGemma2ThemeEmbedding(BaseThemeEmbedding):
    __tablename__ = "bge_multilingual_gemma2_theme_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(3584), nullable=False)
    # __table_args__ = (
    #     Index(
    #         "ix_bge_multilingual_gemma2_theme_embeddings_embedding",
    #         "embeddings",
    #         postgresql_using="hnsw",
    #         postgresql_ops={"embeddings": "public.vector_cosine_ops"},
    #     ),
    # )


class OpenAITextEmbedding3LargeThemeEmbedding(BaseThemeEmbedding):
    __tablename__ = "openai_text_embedding_3_large_theme_embeddings"

    embeddings: Mapped[Embeddings] = mapped_column(Vector(3072), nullable=False)
    # TODO: Same as BGE above
