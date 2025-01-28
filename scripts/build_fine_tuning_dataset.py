#!/usr/bin/env python

import json
import logging
import logging.config
from typing import Any

from sqlalchemy import func, select

from tca.database.models import (
    BGEMultilingualGemma2ChunkEmbedding,
    BGEMultilingualGemma2ThemeEmbedding,
    ChunkEmbeddingBase,
    DocumentChunk,
    Theme,
)
from tca.database.session_manager import PostgresSessionManager

logging.config.fileConfig("logging.conf")
NB_CHUNKS_TO_RETRIEVE = 6
OUTPUT_FILE = "fine_tuning_data/gemma2/theme_to_found_chunks.json"


def query_chunks_by_theme(
    session: Any,
    embedding_cls: Any,
    theme_embeddings: Any,
    nb_chunks_to_retrieve: int,
    min_cos_dist_threshold: float,
) -> list[tuple[str, str, list[str], list[float]]]:
    """
    Query the database for chunks matching a given theme's embeddings.
    Returns a list of tuples containing document ID, document name, chunk texts, and cosine distances.
    """
    chunk_select_query = (
        select(
            DocumentChunk,
            embedding_cls.embeddings.cosine_distance(theme_embeddings).label(
                "cos_distance"
            ),
            func.row_number()
            .over(
                partition_by=DocumentChunk.document_id,
                order_by=embedding_cls.embeddings.cosine_distance(theme_embeddings),
            )
            .label("row_number"),
        )
        .join(
            embedding_cls,
            embedding_cls.chunk_id == DocumentChunk.id,
        )
        .filter(
            embedding_cls.embeddings.cosine_distance(theme_embeddings)
            < min_cos_dist_threshold
        )
        .subquery()
    )

    query = (
        select(
            chunk_select_query.c.document_id,
            func.min(chunk_select_query.c.document_name).label("document_name"),
            func.array_agg(chunk_select_query.c.chunk_text).label("chunk_texts"),
            func.array_agg(chunk_select_query.c.cos_distance).label("cos_distances"),
        )
        .filter(chunk_select_query.c.row_number <= nb_chunks_to_retrieve)
        .group_by(chunk_select_query.c.document_id)
        .order_by(func.min(chunk_select_query.c.cos_distance))
    )

    return session.execute(query).all()


# Second functionality: find closest chunks for themes
def find_closest_chunks_for_themes(
    themes_with_embeddings: list[tuple[Theme, Any]],
    session: Any,
    nb_chunks_to_retrieve: int,
    min_cos_dist_threshold: float,
    chunk_embedding_cls: type[ChunkEmbeddingBase],
) -> dict[str, list[tuple[str, str, list[str], list[float]]]]:
    theme_to_doc_to_chunks = {}
    for theme_info, theme_embeddings in themes_with_embeddings:
        theme_n2 = theme_info.themes[-1]
        results = query_chunks_by_theme(
            session=session,
            embedding_cls=chunk_embedding_cls,
            theme_embeddings=theme_embeddings,
            nb_chunks_to_retrieve=nb_chunks_to_retrieve,
            min_cos_dist_threshold=min_cos_dist_threshold,
        )
        theme_to_doc_to_chunks[theme_n2] = results
    return theme_to_doc_to_chunks


def main():
    logging.info("Starting to find closest chunks for themes.")
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    chunk_embedding_cls = BGEMultilingualGemma2ChunkEmbedding
    theme_embedding_cls = BGEMultilingualGemma2ThemeEmbedding

    themes_with_embeddings_query = select(
        Theme,
        theme_embedding_cls.embeddings,
    ).join(Theme, theme_embedding_cls.theme_id == Theme.id)

    themes_with_embeddings = [
        (row[0], row[1]) for row in session.execute(themes_with_embeddings_query).all()
    ]
    theme_to_doc_to_chunks = find_closest_chunks_for_themes(
        themes_with_embeddings=themes_with_embeddings,
        session=session,
        nb_chunks_to_retrieve=NB_CHUNKS_TO_RETRIEVE,
        min_cos_dist_threshold=0.7,
        chunk_embedding_cls=chunk_embedding_cls,
    )
    logging.info("Successfully found closest chunks for themes.")

    # Initialize an empty dictionary to store the found chunks for each theme
    theme_to_found_chunks = {}

    # Iterate over each theme and its corresponding document chunks
    for theme2, doc_to_chunks in theme_to_doc_to_chunks.items():
        # Initialize the positive and negative lists for each theme
        theme_to_found_chunks[theme2] = {"positive": [], "negative": []}

        # dictionary to store unique chunks along with their cosine distances
        unique_chunks = {}

        # Iterate over each document and its chunks
        for _doc_id, _doc_name, chunks, cos_dists in doc_to_chunks:
            # Update the unique_chunks dictionary with the chunk text and cosine distance
            for chunk, cos_dist in zip(chunks, cos_dists):
                if chunk not in unique_chunks:
                    unique_chunks[chunk] = cos_dist

        # Sort the unique chunks based on their cosine distance
        sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1])

        # Add the top 10 sorted chunks text to the negative list of the theme
        theme_to_found_chunks[theme2]["negative"] = [
            chunk for chunk, _ in sorted_chunks[:10]
        ]

    json.dump(
        theme_to_found_chunks,
        open(OUTPUT_FILE, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    logging.info(
        f"Successfully saved theme to found chunks mapping to JSON file.\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
