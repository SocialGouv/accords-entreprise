#!/usr/bin/env python
import logging
import logging.config
import os
from pathlib import Path

import pandas as pd

from scripts.ingest_themes import ThemeProcessor
from tca.constants import DATA_FOLDER
from tca.database.document_chunk_db_client import (
    DocumentChunkDBClient,
)
from tca.database.models import OllamaBgeM3ChunkEmbedding, OllamaBgeM3ThemeEmbedding
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient
from tca.embedding.embedding_clients import OllamaEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    embedding_client = OllamaEmbeddingClient()
    theme_db_client = ThemeDBClient(
        session=session,
        embedding_client=embedding_client,
        db_theme_prompt_embedding_cls=OllamaBgeM3ThemeEmbedding,
    )
    document_chunk_db_client = DocumentChunkDBClient(
        session=session,
        db_embedding_model_cls=OllamaBgeM3ChunkEmbedding,
    )
    theme_processor = ThemeProcessor(
        embedding_client=embedding_client,
    )
    themes_with_embeddings = theme_db_client.get_themes_with_embeddings()

    found_match_for_themes = []
    for theme in themes_with_embeddings:
        semantic_search_results = document_chunk_db_client.find_matching_documents(
            query_embeddings=theme.theme_prompt_embedding,
            cos_dist_threshold=0.5,
        )
        if semantic_search_results:
            found_match_for_themes.append(
                {
                    "themes": theme.themes,
                    "semantic_search_results": semantic_search_results,
                }
            )

    theme_assignment_df = theme_processor.build_theme_assignment_df(
        found_match_for_themes
    )
    if theme_assignment_df.empty:
        logger.warning("No themes were assigned to documents.")
        return
    expected_df_path = Path(os.path.join(DATA_FOLDER, "normalized_themes.xlsx"))
    expected_df = pd.read_excel(expected_df_path).map(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    comparison_df, theme_performance_df = theme_processor.compare_with_expected(
        theme_assignment_df, expected_df
    )
    output_path = os.path.join(DATA_FOLDER, "comparison_results.xlsx")
    theme_processor.save_theming_results(
        comparison_df, theme_performance_df, output_path
    )


if __name__ == "__main__":
    main()
