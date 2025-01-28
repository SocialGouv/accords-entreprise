import json
import logging
import logging.config
import re
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from tca.database.document_chunk_db_client import (
    BaseDocumentChunkDBClient,
    DocumentChunkDBClient,
)
from tca.database.theme_db_client import ThemeWithEmbedding

logging.config.fileConfig("logging.conf")

NB_CHUNKS_TO_RETRIEVE_PER_DOC = 6


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
                # print(theme.themes[-1])
                # for result in semantic_search_results:
                #     print(
                #         f"distance: {result['distance']}\n\tchunk_text: {result['chunk'].chunk_text}"
                #     )
                found_match_for_themes.append(
                    {
                        "themes": theme.themes,
                        "semantic_search_results": semantic_search_results,
                    }
                )

        return EmbeddingOnlyThemeAssignmentStrategy._build_theme_assignment_df(
            found_match_for_themes
        )


class EmbeddingLLMThemeAssignmentStrategy:
    def __init__(
        self,
        session: Any,
        llm_client: Any,
        doc_chunk_db_client: BaseDocumentChunkDBClient,
        min_cos_dist_threshold: float = 0.75,
        nb_chunks_to_retrieve_per_doc: int = NB_CHUNKS_TO_RETRIEVE_PER_DOC,
    ):
        super().__init__()
        self.session = session
        self.llm_client = llm_client
        self.doc_chunk_db_client = doc_chunk_db_client
        self.min_cos_dist_threshold = min_cos_dist_threshold
        self.nb_chunks_to_retrieve_per_doc = nb_chunks_to_retrieve_per_doc

    def _process_chunk_results(
        self, results: list[tuple[str, str, list[str], list[float]]]
    ) -> dict[str, set]:
        """
        Processes query results and organizes them by document name prefix.
        """
        doc_sentences: dict[str, set] = {}
        for _document_id, document_name, chunk_texts, _cos_distances in results:
            document_id_match = re.match(r"[AT\d]+", document_name)
            document_name_prefix = (
                document_id_match.group(0) if document_id_match else "Unknown"
            ).lower()
            doc_sentences.setdefault(document_name_prefix, set()).update(chunk_texts)
        return doc_sentences

    def find_matching_themes_for_documents(
        self,
        themes_with_embeddings: list[ThemeWithEmbedding],
    ) -> pd.DataFrame:
        """
        For each theme in the list, query the database for chunks matching the theme's embeddings.
        Then, for each document, prompt the LLM to determine if the theme is mentioned in the document.
        """
        theme_descriptions = []
        possible_themes = set()
        theme2_to_theme_info = {}

        doc_sentences: dict[str, set] = {}
        for theme_info in themes_with_embeddings:
            theme = theme_info["theme"]
            theme_embeddings = theme_info["embeddings"]
            theme_n2 = theme.themes[-1]
            possible_themes.add(theme_n2)
            theme2_to_theme_info[theme_n2] = theme
            theme_descriptions.append(f'"{theme_n2}" : "{theme.description}"')

            results = self.doc_chunk_db_client.query_chunks_by_theme(
                theme_embeddings=theme_embeddings,
                min_cos_dist_threshold=self.min_cos_dist_threshold,
                nb_chunks_to_retrieve=self.nb_chunks_to_retrieve_per_doc,
            )

            for _document_id, document_name, chunk_texts, _cos_distances in results:
                document_id_match = re.match(r"[AT\d]+", document_name)
                document_name_prefix = (
                    document_id_match.group(0) if document_id_match else "Unknown"
                ).lower()
                doc_sentences.setdefault(document_name_prefix, set()).update(
                    chunk_texts
                )

            themes_str = "\n- ".join(theme_descriptions)

        extracted_results = []
        for document_name_prefix, sentences in doc_sentences.items():
            sentences_str = "\n- ".join(f'"{text}"' for text in sentences)
            prompt = f"""
Je souhaite déterminer si les thèmes suivants sont abordés explicitement dans un accord d'entreprise.
Les thèmes sont les suivants, avec la forme "theme : description du thème" :
- {themes_str}

Ces phrases proviennent de l'accord d'entreprise :

- {sentences_str}

Garde chaque thème qui est abordé explicitement en positif ou en négatif dans au moins une des phrases et ignore les autres thèmes.
Retourne le résultat sous forme de JSON sans balises de bloc de code avec le format suivant :

["theme1", "theme2", ...]
            """
            response = self.llm_client.generate_text(prompt).strip()
            if response.startswith("```json"):
                response = response[7:-3].strip()
            response_obj = json.loads(response)
            for theme_n2 in response_obj:
                best_matching_theme = max(
                    possible_themes, key=lambda t: fuzz.ratio(theme_n2, t)
                )
                extracted_results.append(
                    {
                        "Document": document_name_prefix,
                        "Thème n1": theme2_to_theme_info[best_matching_theme].themes[0],
                        "Thème n2": theme2_to_theme_info[best_matching_theme].themes[1],
                    }
                )
        return pd.DataFrame(extracted_results)
