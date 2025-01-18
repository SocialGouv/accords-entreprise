#!/usr/bin/env python
import logging
import logging.config
import os
from pathlib import Path

import pandas as pd

from scripts.ingest_themes import ThemeProcessor
from tca.constants import DATA_FOLDER
from tca.database.models import (
    OpenAITextEmbedding3LargeThemeEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient
from tca.text.llm_clients import OpenAIAPIClient
from tca.theme_assignment_strategies import (
    EmbeddingLLMThemeAssignmentStrategy,
)

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    # document_chunk_db_client = DocumentChunkDBClient(
    #     session=session,
    #     db_embedding_model_cls=OpenAITextEmbedding3LargeChunkEmbedding,
    # )
    # retrieval_strategy = EmbeddingOnlyThemeAssignmentStrategy(document_chunk_db_client = document_chunk_db_client, cos_dist_threshold=0.5)

    llm_client = OpenAIAPIClient(
        model_name=os.environ["OPENAI_LLM_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    retrieval_strategy = EmbeddingLLMThemeAssignmentStrategy(
        session=session, llm_client=llm_client, nb_chunks_to_retrieve=6
    )

    theme_db_client = ThemeDBClient(
        session=session,
        db_theme_prompt_embedding_cls=OpenAITextEmbedding3LargeThemeEmbedding,
    )

    themes_with_embeddings = theme_db_client.get_themes_with_their_embeddings()
    theme_assignment_df = retrieval_strategy.find_matching_themes_for_documents(
        themes_with_embeddings
    )

    if theme_assignment_df.empty:
        logger.warning("No themes were assigned to documents.")
        return
    expected_df_path = Path(os.path.join(DATA_FOLDER, "normalized_themes.xlsx"))
    expected_df = pd.read_excel(expected_df_path).map(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    per_theme_metrics_df, overall_metrics_df = ThemeProcessor.evaluate_classification(
        theme_assignment_df, expected_df
    )
    output_path = os.path.join(DATA_FOLDER, "comparison_results.xlsx")
    ThemeProcessor.save_theming_results(
        per_theme_metrics_df, overall_metrics_df, output_path
    )


if __name__ == "__main__":
    main()
