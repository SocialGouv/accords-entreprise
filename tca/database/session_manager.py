import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool


class PostgresSessionManager:
    @property
    def session(self) -> Session:
        return self._session

    def __init__(self):
        self.postgres_user = os.getenv("POSTGRES_USER")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD")
        self.postgres_db = os.getenv("POSTGRES_DB")
        self.engine = create_engine(
            f"postgresql://{self.postgres_user}:{self.postgres_password}@localhost/{self.postgres_db}",
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
        )
        Session = sessionmaker(bind=self.engine)

        self._session = Session()
        self._session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    def __del__(self):
        self._session.close()
        self.engine.dispose()

    def full_reset_chunks(self) -> None:
        self._session.execute(text("TRUNCATE TABLE document_chunks CASCADE"))
        self._session.execute(
            text("ALTER SEQUENCE document_chunks_id_seq RESTART WITH 1")
        )
        self._session.execute(
            text("ALTER SEQUENCE bge_m3_chunk_embeddings_id_seq RESTART WITH 1")
        )
        self._session.execute(
            text(
                "ALTER SEQUENCE bge_multilingual_gemma2_chunk_embeddings_id_seq RESTART WITH 1"
            )
        )
        self._session.execute(
            text(
                "ALTER SEQUENCE openai_text_embedding_3_large_chunk_embeddings_id_seq RESTART WITH 1"
            )
        )

    def full_reset_themes(self) -> None:
        self._session.execute(text("TRUNCATE TABLE themes CASCADE"))
        self._session.execute(text("ALTER SEQUENCE themes_id_seq RESTART WITH 1"))
        self._session.execute(
            text("ALTER SEQUENCE bge_m3_theme_embeddings_id_seq RESTART WITH 1")
        )
        self._session.execute(
            text(
                "ALTER SEQUENCE bge_multilingual_gemma2_theme_embeddings_id_seq RESTART WITH 1"
            )
        )
        self._session.execute(
            text(
                "ALTER SEQUENCE openai_text_embedding_3_large_theme_embeddings_id_seq RESTART WITH 1"
            )
        )
