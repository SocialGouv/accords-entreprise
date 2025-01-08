#!/usr/bin/env python
import glob
import logging
import logging.config
from pathlib import Path

from tca.constants import DATA_FOLDER
from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.document_ingester import DocumentIngester
from tca.embedding.embedding_clients import OllamaEmbeddingClient
from tca.text.chunker import DelimiterChunker, SemanticChunker

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


INPUT_FILE_PREFIXES = [
    "T05123005651",
    "A05114000445",
    "T09224067466",
    "T00624061516",
    "T07624061950",
    "T04524061140",
    "T08722002525",
    "T01023002490",
    "T00624061500",
    "T09524061379",
    "T03624060237",
    "T07424061122",
    "T09224067135",
    "T09423011056",
    "T03424061573",
    "T01424061178",
    "T00624061455",
    "T06024060960",
    "T08424060468",
    "T20A23060010",
    "T09023001809",
    "T09523006784",
    "T01423060053",
    "T09219008191",
]


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    postgres_session_manager.full_reset()
    session = postgres_session_manager.session

    embedding_client = OllamaEmbeddingClient()
    ollama_bge_m3_config = DocumentChunkManagerConfig(
        embedding_client=embedding_client,
        db_embedding_model_cls=OllamaBgeM3Embedding,
    )

    document_chunk_manager = DocumentChunkManager(
        session,
        ollama_bge_m3_config,
        chunker=SemanticChunker(
            embedding_client=embedding_client,
            pre_chunker=DelimiterChunker(),
            min_chunk_size=100,
            similarity_threshold=0.9,
        ),
        # chunker=ParagraphChunker(),
        # chunker=RecursiveChunker(chunk_size=100, chunk_overlap=0),
    )

    documents_folder = f"{DATA_FOLDER}/accords_entreprise_niveau2"
    document_paths: list[Path] = []
    for prefix in INPUT_FILE_PREFIXES:
        document_paths.extend(
            Path(file) for file in glob.glob(f"{documents_folder}/{prefix}*")
        )

    doc_ingester = DocumentIngester(document_chunk_manager)
    doc_ingester.ingest_documents(document_paths=document_paths)


if __name__ == "__main__":
    main()
