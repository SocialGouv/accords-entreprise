#!/usr/bin/env python
import logging
import logging.config
import os
from pathlib import Path

from tca.constants import DATA_FOLDER
from tca.database.models import (
    OllamaBgeM3ThemeEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient
from tca.embedding.embedding_clients import OllamaEmbeddingClient
from tca.theme_processor import ThemeProcessor

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    postgres_session_manager.full_reset_themes()
    session = postgres_session_manager.session

    embedding_client = OllamaEmbeddingClient()

    theme_db_client = ThemeDBClient(
        session=session,
        embedding_client=embedding_client,
        db_theme_prompt_embedding_cls=OllamaBgeM3ThemeEmbedding,
    )

    theme_manager = ThemeProcessor(
        embedding_client=embedding_client,
    )

    theme_list_path = Path(os.path.join(DATA_FOLDER, "theme_list.csv"))
    themes = theme_manager.load_themes(theme_list_path)
    theme_db_client.ingest_themes_in_db(themes)


if __name__ == "__main__":
    main()
