#!/usr/bin/env python
import json
import logging
import logging.config
import os
import re
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from tca.constants import DATA_FOLDER
from tca.database.document_chunk_db_client import (
    BaseDocumentChunkDBClient,
    DocumentChunkDBClient,
)
from tca.database.models import (
    BGEMultilingualGemma2ChunkEmbedding,
    BGEMultilingualGemma2ThemeEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.database.theme_db_client import ThemeDBClient, ThemeWithEmbedding
from tca.text.llm_clients import OpenAIAPIClient

logging.config.fileConfig("logging.conf")

NB_CHUNKS_TO_RETRIEVE_PER_DOC = 6


class LLMThemingAnalyser:
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
        Retiens les phrases qui t'ont aidé à prendre ta décision.
        Retourne le résultat sous forme de JSON sans balises de bloc de code avec le format suivant :

        {{"theme1": ["Phrase qui a permis de prendre cette décision 1", "Phrase qui a permis de prendre cette décision 2"], "theme2": [], ...}}
            """
            # logging.info(f"Prompt: {prompt}\n")
            response = self.llm_client.generate_text(prompt).strip()
            # logging.info(f"Response: {response}\n")
            if response.startswith("```json"):
                response = response[7:-3].strip()
            try:
                response_obj = json.loads(response)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to decode JSON response: {e}")
                response_obj = {}
            for theme_n2, sentences in response_obj.items():
                best_match = max(possible_themes, key=lambda t: fuzz.ratio(theme_n2, t))
                theme_n2 = best_match
                for sentence in sentences:
                    extracted_results.append(
                        {
                            "Document": document_name_prefix,
                            "Thème (N1)": theme2_to_theme_info[theme_n2].themes[0],
                            "Thème (N2)": theme2_to_theme_info[theme_n2].themes[1],
                            "Phrase": sentence,
                        }
                    )
        return pd.DataFrame(extracted_results)


def main():
    OUTPUT_FILE = os.path.join(DATA_FOLDER, "themes_found_with_llm_and_why.xlsx")
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    chunk_embedding_cls = BGEMultilingualGemma2ChunkEmbedding
    theme_embedding_cls = BGEMultilingualGemma2ThemeEmbedding
    theme_db_client = ThemeDBClient(
        session=session,
        theme_embedding_cls=theme_embedding_cls,
    )

    document_chunk_db_client = DocumentChunkDBClient(
        session=session,
        chunk_embedding_cls=chunk_embedding_cls,
    )
    llm_client = OpenAIAPIClient(
        model_name=os.environ["OPENAI_LLM_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    llm_theming_analyser = LLMThemingAnalyser(
        session=session,
        llm_client=llm_client,
        doc_chunk_db_client=document_chunk_db_client,
        min_cos_dist_threshold=0.5,
        nb_chunks_to_retrieve_per_doc=NB_CHUNKS_TO_RETRIEVE_PER_DOC,
    )

    themes_with_embeddings = theme_db_client.get_themes_with_their_embeddings()
    logging.info(
        "Retrieved %d themes with their embeddings.", len(themes_with_embeddings)
    )

    llm_theming_result_df = llm_theming_analyser.find_matching_themes_for_documents(
        themes_with_embeddings=themes_with_embeddings
    )

    llm_theming_result_df.to_excel(OUTPUT_FILE, index=False)
    logging.info(f"Saved theme assignment results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
