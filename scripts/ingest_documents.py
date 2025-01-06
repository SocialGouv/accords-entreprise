#!/usr/bin/env python

import logging
import logging.config

from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.document_ingester import DocumentIngester
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

    doc_ingester = DocumentIngester(document_chunk_manager)
    doc_ingester.ingest_documents()


if __name__ == "__main__":
    main()
