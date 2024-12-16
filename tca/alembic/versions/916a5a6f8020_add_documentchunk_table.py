"""Add DocumentChunk table

Revision ID: 916a5a6f8020
Revises:
Create Date: 2024-12-16 22:01:06.331877

"""

from alembic import op
from sqlalchemy import ARRAY, JSON, Column, Float, Integer, String, Text

revision = "916a5a6f8020"  # pragma: allowlist secret
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create table
    op.create_table(
        "document_chunks",
        Column("id", Integer, primary_key=True),
        Column("document_id", String, nullable=False),
        Column("chunk_text", Text, nullable=False),
        Column("embedding", ARRAY(Float), nullable=False),  # Use ARRAY for PostgreSQL
        Column("metadata", JSON),  # JSON-compatible across DBs
    )


def downgrade() -> None:
    # Drop table
    op.drop_table("document_chunks")
