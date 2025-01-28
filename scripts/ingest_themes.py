#!/usr/bin/env python
import os
from pathlib import Path

from tca.constants import DATA_FOLDER
from tca.database.models import (
    BGEMultilingualGemma2ThemeEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient
from tca.embedding.embedding_clients import (
    OpenAIEmbeddingClient,
)
from tca.theme_processor import ThemeProcessor


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    postgres_session_manager.full_reset_themes()
    session = postgres_session_manager.session

    # embedding_client = OllamaEmbeddingClient()
    # embedding_client = FlagLLMEmbeddingClient()
    # embedding_client = OpenAIEmbeddingClient()
    embedding_client = OpenAIEmbeddingClient(
        model=os.environ["SCALEWAY_MODEL_NAME"],
        api_key=os.environ["SCALEWAY_API_KEY"],
        base_url=os.environ["SCALEWAY_BASE_URL"],
    )

    theme_db_client = ThemeDBClient(
        session=session,
        # db_theme_prompt_embedding_cls=OpenAITextEmbedding3LargeThemeEmbedding,
        theme_embedding_cls=BGEMultilingualGemma2ThemeEmbedding,
    )

    theme_list_path = Path(os.path.join(DATA_FOLDER, "theme_list.csv"))
    themes = ThemeProcessor.load_themes(theme_list_path)
    theme_db_client.ingest_themes_in_db(
        embedding_client=embedding_client, themes=themes
    )


if __name__ == "__main__":
    main()
