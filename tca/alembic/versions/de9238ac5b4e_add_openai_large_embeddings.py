"""Add OpenAI large embeddings

Revision ID: de9238ac5b4e
Revises: 4fcdf13a2faf
Create Date: 2025-01-15 15:01:22.394347

"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy.vector import VECTOR

# revision identifiers, used by Alembic.
revision = "de9238ac5b4e"
down_revision = "4fcdf13a2faf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "openai_text_embedding_3_large_chunk_embeddings",
        sa.Column("embeddings", VECTOR(dim=3072), nullable=False),
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
        "openai_text_embedding_3_large_theme_embeddings",
        sa.Column("embeddings", VECTOR(dim=3072), nullable=False),
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
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_table("openai_text_embedding_3_large_theme_embeddings")
    op.drop_table("openai_text_embedding_3_large_chunk_embeddings")
