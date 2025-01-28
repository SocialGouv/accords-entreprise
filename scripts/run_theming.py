#!/usr/bin/env python
import logging
import logging.config
import os
from pathlib import Path

import pandas as pd

from scripts.ingest_themes import ThemeProcessor
from tca.constants import DATA_FOLDER
from tca.database.document_chunk_db_client import DocumentChunkDBClient
from tca.database.models import (
    BGEMultilingualGemma2ChunkEmbedding,
    BGEMultilingualGemma2ThemeEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient
from tca.theme_assignment_strategies import (
    EmbeddingOnlyThemeAssignmentStrategy,
)

logging.config.fileConfig("logging.conf")


def main() -> None:
    logging.info("Starting theme assignment process.")
    OUTPUT_FILE = os.path.join(
        DATA_FOLDER, "comparison_results_embeddings_only_gemma2.xlsx"
    )
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    chunk_embedding_cls = BGEMultilingualGemma2ChunkEmbedding
    theme_embedding_cls = BGEMultilingualGemma2ThemeEmbedding

    document_chunk_db_client = DocumentChunkDBClient(
        session=session,
        chunk_embedding_cls=chunk_embedding_cls,
    )
    # llm_client = OpenAIAPIClient(
    #     model_name=os.environ["OPENAI_LLM_MODEL"],
    #     api_key=os.environ["OPENAI_API_KEY"],
    # )

    retrieval_strategy = EmbeddingOnlyThemeAssignmentStrategy(
        document_chunk_db_client=document_chunk_db_client, cos_dist_threshold=0.4
    )
    # retrieval_strategy = EmbeddingLLMThemeAssignmentStrategy(
    #     session=session,
    #     llm_client=llm_client,
    #     doc_chunk_db_client=document_chunk_db_client,
    #     min_cos_dist_threshold=0.75,
    #     nb_chunks_to_retrieve=6,
    # )

    theme_db_client = ThemeDBClient(
        session=session,
        theme_embedding_cls=theme_embedding_cls,
    )

    logging.info("Retrieving themes with their embeddings.")
    themes_with_embeddings = theme_db_client.get_themes_with_their_embeddings()
    logging.info(
        "Retrieved %d themes with their embeddings.", len(themes_with_embeddings)
    )
    logging.info("Assigning themes to documents.")
    theme_assignment_df = retrieval_strategy.find_matching_themes_for_documents(
        themes_with_embeddings
    )
    logging.info("Finished assigning themes to documents.")

    if theme_assignment_df.empty:
        logging.warning("No themes were assigned to documents.")
        return
    expected_df_path = Path(os.path.join(DATA_FOLDER, "normalized_themes.xlsx"))
    expected_df = pd.read_excel(expected_df_path).map(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    per_theme_metrics_df, overall_metrics_df = ThemeProcessor.evaluate_classification(
        theme_assignment_df, expected_df
    )
    ThemeProcessor.save_theming_results(
        per_theme_metrics_df, overall_metrics_df, OUTPUT_FILE
    )
    logging.info(
        f"Saved theme assignment results to {OUTPUT_FILE}",
    )


if __name__ == "__main__":
    main()
