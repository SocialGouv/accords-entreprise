#!/usr/bin/env python

import logging
import logging.config

from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.embedding.embedding_clients import OllamaEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


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

    # query_embeddings = embedding_client.embed(["Durée collective du temps de travail"])[0]
    query_embeddings = embedding_client.embed(
        ["Organisation de la durée et de l'aménagement du temps de travail"]
    )[0]
    results = document_chunk_manager.query_similar_chunks(
        query_embeddings=query_embeddings, cos_dist_threshold=0.4
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
