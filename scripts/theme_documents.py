#!/usr/bin/env python
import logging
import logging.config
import os

import pandas as pd

from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.document_ingester import DATA_FOLDER
from tca.embedding.embedding_clients import OllamaEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")


# TODO: This early test should show that the pipeline works. Store the result so it can be compared to our labeled data
# TODO: Finally, implement the real embedding and chunking logic using a good embedder and chunking with langchain or similar
def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    embedding_client = OllamaEmbeddingClient()
    ollama_bge_m3_config = DocumentChunkManagerConfig(
        embedding_client=embedding_client,
        db_embedding_model_cls=OllamaBgeM3Embedding,
    )

    document_chunk_manager = DocumentChunkManager(
        session,
        ollama_bge_m3_config,
    )

    theme_list_path = os.path.join(DATA_FOLDER, "theme_list.csv")

    themes = pd.read_csv(theme_list_path)
    results = []
    for _index, theme in themes.iterrows():
        # prompt = (
        #     f"Generate an embedding for a theme specifically related to company agreements in France. "
        #     f"The theme hierarchy, from broader to narrower, is: {theme['niveau 1']} -> {theme['niveau 2']}. "
        #     f"Focus the embedding on representing the semantic meaning of this theme in the context of French labor law and company agreements."
        # )
        prompt = f"{theme['niveau 1']} -> {theme['niveau 2']}"

        query_embeddings = embedding_client.embed([prompt])[0]
        semantic_search_results = document_chunk_manager.query_similar_chunks(
            query_embeddings=query_embeddings, cos_dist_threshold=0.4, top_k=1
        )
        if semantic_search_results:
            results.append(
                {
                    "themes": [theme["niveau 1"], theme["niveau 2"]],
                    "semantic_search_results": semantic_search_results,
                }
            )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
