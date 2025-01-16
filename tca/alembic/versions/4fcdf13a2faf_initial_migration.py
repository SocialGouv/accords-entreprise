"""Initial migration

Revision ID: 4fcdf13a2faf
Revises:
Create Date: 2025-01-13 19:12:54.124300

"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy.vector import VECTOR

# revision identifiers, used by Alembic.
revision = "4fcdf13a2faf"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("document_name", sa.String(), nullable=False),
        sa.Column("chunk_text", sa.String(), nullable=False),
        sa.Column("extra_metadata", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("UP_TO_DATE", "OUTDATED", "DELETED", native_enum=False),
            nullable=False,
        ),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "themes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("themes", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("UP_TO_DATE", "OUTDATED", "DELETED", native_enum=False),
            nullable=False,
        ),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "bge_m3_chunk_embeddings",
        sa.Column("embeddings", VECTOR(dim=1024), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_bge_m3_chunk_embeddings_embedding",
        "bge_m3_chunk_embeddings",
        ["embeddings"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embeddings": "public.vector_cosine_ops"},
    )
    op.create_table(
        "bge_m3_theme_embeddings",
        sa.Column("embeddings", VECTOR(dim=1024), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("theme_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["theme_id"],
            ["themes.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_bge_m3_theme_embeddings_embedding",
        "bge_m3_theme_embeddings",
        ["embeddings"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embeddings": "public.vector_cosine_ops"},
    )
    op.create_table(
        "bge_multilingual_gemma2_chunk_embeddings",
        sa.Column("embeddings", VECTOR(dim=3584), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "bge_multilingual_gemma2_theme_embeddings",
        sa.Column("embeddings", VECTOR(dim=3584), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("theme_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["theme_id"],
            ["themes.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("bge_multilingual_gemma2_theme_embeddings")
    op.drop_table("bge_multilingual_gemma2_chunk_embeddings")
    op.drop_index(
        "ix_bge_m3_theme_embeddings_embedding",
        table_name="bge_m3_theme_embeddings",
        postgresql_using="hnsw",
        postgresql_ops={"embeddings": "public.vector_cosine_ops"},
    )
    op.drop_table("bge_m3_theme_embeddings")
    op.drop_index(
        "ix_bge_m3_chunk_embeddings_embedding",
        table_name="bge_m3_chunk_embeddings",
        postgresql_using="hnsw",
        postgresql_ops={"embeddings": "public.vector_cosine_ops"},
    )
    op.drop_table("bge_m3_chunk_embeddings")
    op.drop_table("themes")
    op.drop_table("document_chunks")
