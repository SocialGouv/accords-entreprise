import logging
import logging.config
import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from tca.database.document_chunk_db_client import (
    DocumentChunkDBClient,
)
from tca.database.theme_db_client import ThemeWithEmbedding

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class BaseThemeAssignmentStrategy(ABC):
    @abstractmethod
    def find_matching_themes_for_documents(
        self,
        themes_with_embeddings: list[ThemeWithEmbedding],
        document_chunk_db_client: DocumentChunkDBClient,
    ) -> pd.DataFrame:
        raise NotImplementedError


class EmbeddingOnlyThemeAssignmentStrategy(BaseThemeAssignmentStrategy):
    def __init__(self, cos_dist_threshold: float = 0.5):
        super().__init__()
        self.cos_dist_threshold = cos_dist_threshold

    @staticmethod
    def _build_theme_assignment_df(
        theme_assignments: list[dict[str, Any]],
    ) -> pd.DataFrame:
        structured_theme_assignment = []
        for theme_assignment in theme_assignments:
            for search_result in theme_assignment["semantic_search_results"]:
                chunk = search_result["chunk"]
                document_id_match = re.match(r"[AT\d]+", chunk.document_name)
                document_id = (
                    document_id_match.group(0) if document_id_match else "Unknown"
                )
                document_id = document_id.lower()
                structured_theme_assignment.append(
                    {
                        "Document ID": document_id,
                        "Thème n1": theme_assignment["themes"][0],
                        "Thème n2": theme_assignment["themes"][1],
                        "Distance": search_result["distance"],
                        "Chunk": chunk.chunk_text,
                    }
                )
        return pd.DataFrame(structured_theme_assignment)

    def find_matching_themes_for_documents(
        self,
        themes_with_embeddings: list[ThemeWithEmbedding],
        document_chunk_db_client: DocumentChunkDBClient,
    ) -> pd.DataFrame:
        found_match_for_themes: list[dict] = []
        for theme_with_embeddings in themes_with_embeddings:
            theme = theme_with_embeddings["theme"]
            semantic_search_results = (
                document_chunk_db_client.find_matching_documents_with_cos_dist(
                    query_embeddings=theme_with_embeddings["embeddings"],
                    cos_dist_threshold=self.cos_dist_threshold,
                )
            )
            if semantic_search_results:
                print(theme.themes[-1])
                for result in semantic_search_results:
                    print(
                        f"distance: {result['distance']}\n\tchunk_text: {result['chunk'].chunk_text}"
                    )
                found_match_for_themes.append(
                    {
                        "themes": theme.themes,
                        "semantic_search_results": semantic_search_results,
                    }
                )

        return EmbeddingOnlyThemeAssignmentStrategy._build_theme_assignment_df(
            found_match_for_themes
        )
