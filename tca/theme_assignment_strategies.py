import json
import logging
import logging.config
import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy import func, select

from tca.database.document_chunk_db_client import (
    DocumentChunkDBClient,
)
from tca.database.models import (
    DocumentChunk,
    OpenAITextEmbedding3LargeChunkEmbedding,
)
from tca.database.theme_db_client import ThemeWithEmbedding

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

NB_CHUNKS_TO_RETRIEVE = 6


class BaseThemeAssignmentStrategy(ABC):
    @abstractmethod
    def find_matching_themes_for_documents(
        self,
        themes_with_embeddings: list[ThemeWithEmbedding],
    ) -> pd.DataFrame:
        raise NotImplementedError


class EmbeddingOnlyThemeAssignmentStrategy(BaseThemeAssignmentStrategy):
    def __init__(
        self,
        document_chunk_db_client: DocumentChunkDBClient,
        cos_dist_threshold: float = 0.5,
    ):
        super().__init__()
        self.cos_dist_threshold = cos_dist_threshold
        self.document_chunk_db_client = document_chunk_db_client

    @staticmethod
    def _build_theme_assignment_df(
        theme_assignments: list[dict[str, Any]],
    ) -> pd.DataFrame:
        structured_theme_assignment = []
        for theme_assignment in theme_assignments:
            for search_result in theme_assignment["semantic_search_results"]:
                chunk = search_result["chunk"]
                document_name_prefix_match = re.match(r"[AT\d]+", chunk.document_name)
                document_prefix_name = (
                    document_name_prefix_match.group(0)
                    if document_name_prefix_match
                    else "Unknown"
                )
                document_prefix_name = document_prefix_name.lower()
                structured_theme_assignment.append(
                    {
                        "Document": document_prefix_name,
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
    ) -> pd.DataFrame:
        found_match_for_themes: list[dict] = []
        for theme_with_embeddings in themes_with_embeddings:
            theme = theme_with_embeddings["theme"]
            semantic_search_results = (
                self.document_chunk_db_client.find_matching_documents_with_cos_dist(
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


class EmbeddingLLMThemeAssignmentStrategy(BaseThemeAssignmentStrategy):
    def __init__(
        self,
        session: Any,
        llm_client: Any,
        nb_chunks_to_retrieve: int = NB_CHUNKS_TO_RETRIEVE,
    ):
        super().__init__()
        self.session = session
        self.llm_client = llm_client
        self.nb_chunks_to_retrieve = nb_chunks_to_retrieve

    def find_matching_themes_for_documents(
        self, themes_with_embeddings: list[ThemeWithEmbedding]
    ) -> pd.DataFrame:
        doc_sentences: dict[str, set[str]] = {}

        theme2_to_theme_info = {}
        theme_descriptions: list[str] = []
        possible_themes = set()
        for theme_info in themes_with_embeddings:
            theme = theme_info["theme"]
            query_embeddings = theme_info["embeddings"]
            theme_n2: str = theme.themes[-1]
            possible_themes.add(theme_n2)
            theme2_to_theme_info[theme_n2] = theme
            theme_descriptions.append(f'"{theme_n2}" : "{theme.description}"')

            chunk_select_query = (
                select(
                    DocumentChunk,
                    OpenAITextEmbedding3LargeChunkEmbedding.embeddings.cosine_distance(
                        query_embeddings
                    ).label("cos_distance"),
                    func.row_number()
                    .over(
                        partition_by=DocumentChunk.document_id,
                        order_by=OpenAITextEmbedding3LargeChunkEmbedding.embeddings.cosine_distance(
                            query_embeddings
                        ),
                    )
                    .label("row_number"),
                )
                .join(
                    OpenAITextEmbedding3LargeChunkEmbedding,
                    OpenAITextEmbedding3LargeChunkEmbedding.chunk_id
                    == DocumentChunk.id,
                )
                .filter(
                    OpenAITextEmbedding3LargeChunkEmbedding.embeddings.cosine_distance(
                        query_embeddings
                    )
                    < 0.75
                )
                .subquery()
            )

            query = (
                select(
                    chunk_select_query.c.document_id,
                    func.min(chunk_select_query.c.document_name).label("document_name"),
                    func.array_agg(chunk_select_query.c.chunk_text).label(
                        "chunk_texts"
                    ),
                    func.array_agg(chunk_select_query.c.cos_distance).label(
                        "cos_distances"
                    ),
                )
                .filter(chunk_select_query.c.row_number <= self.nb_chunks_to_retrieve)
                .group_by(chunk_select_query.c.document_id)
                .order_by(func.min(chunk_select_query.c.cos_distance))
            )

            results = self.session.execute(query).all()

            for _document_id, document_name, chunk_texts, _cos_distances in results:
                document_id_match = re.match(r"[AT\d]+", document_name)
                document_name_prefix = (
                    document_id_match.group(0) if document_id_match else "Unknown"
                )
                document_name_prefix = document_name_prefix.lower()
                doc_sentences.setdefault(document_name_prefix, set()).update(
                    f'"{text}"' for text in chunk_texts
                )

        themes_str: str = "\n- ".join(theme_descriptions)

        results = []
        for document_name_prefix, sentences in doc_sentences.items():
            print(f"Document: {document_name_prefix}")
            sentences_str: str = "\n- ".join(sentences)

            prompt: str = f"""
Je souhaite déterminer si les thèmes suivants sont abordés explicitement dans un accord d'entreprise.
Les thèmes sont les suivants, avec la forme "theme : description du thème" :
- {themes_str}

Ces phrases proviennent de l'accord d'entreprise :

- {sentences_str}

Garde chaque thème qui est abordé explicitement en positif ou en négatif dans au moins une des phrases et ignore les autres thèmes.
Retourne le résultat sous forme de JSON sans balises de bloc de code avec le format suivant :

["theme1", "theme2", ...]
            """
            print(f"prompt {prompt}")
            response: str = self.llm_client.generate_text(prompt)
            response = response.strip()
            if response.startswith("```json"):
                logging.warning(f"{document_name_prefix}: ```json DETECTED")
                response = response[7:-3].strip()
            logging.info(f"Themes ({document_name_prefix}): {response}")
            response_obj = json.loads(response)
            for theme_n2 in response_obj:
                best_matching_theme = max(
                    possible_themes, key=lambda t: fuzz.ratio(theme_n2, t)
                )
                results.append(
                    {
                        "Document": document_name_prefix,
                        "Thème n1": theme2_to_theme_info[best_matching_theme].themes[0],
                        "Thème n2": theme2_to_theme_info[best_matching_theme].themes[1],
                    }
                )
        return pd.DataFrame(results)
