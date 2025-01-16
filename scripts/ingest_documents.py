#!/usr/bin/env python
import glob
from pathlib import Path

from tca.constants import DATA_FOLDER
from tca.database.document_chunk_db_client import (
    DocumentChunkDBClient,
)
from tca.database.models import (
    OpenAITextEmbedding3LargeChunkEmbedding,
)
from tca.database.session_manager import PostgresSessionManager
from tca.embedding.document_ingester import DocumentIngester
from tca.embedding.embedding_clients import (
    OpenAIEmbeddingClient,
)
from tca.text.chunker import DelimiterChunker

INPUT_FILE_PREFIXES = [
    "T00624061455",
    "T00624061500",
    "T00624061516",
    "T01023002490",
    "T01423060053",
    "T01424061178",
    "T03424061573",
    "T03624060237",
    "T04524061140",
    "T05123005651",
    "T06024060960",
    "T07424061122",
    "T07624061950",
    "T08424060468",
    "T08722002525",
    "T09023001809",
    "T09219008191",
    "T09224067135",
    "T09224067466",
    "T09423011056",
    "T09523006784",
    "T09524061379",
    "T20A23060010",
]


def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    postgres_session_manager.full_reset_chunks()
    session = postgres_session_manager.session
    # embedding_client = OllamaEmbeddingClient()
    # scaleway_embedding_client = OpenAIEmbeddingClient(
    #     model=os.environ["SCALEWAY_MODEL_NAME"],
    #     api_key=os.environ["SCALEWAY_API_KEY"],
    #     base_url=os.environ["SCALEWAY_BASE_URL"],
    # )
    openai_embedding_client = OpenAIEmbeddingClient()

    embedding_client = openai_embedding_client
    # chunker = SemanticChunker(
    #     pre_chunker=DelimiterChunker(),
    #     min_chunk_size=100,
    #     similarity_threshold=0.8,
    #     embedding_client=embedding_client,
    # )
    chunker = DelimiterChunker(min_chunk_size=40)
    # chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)

    document_chunk_db_client = DocumentChunkDBClient(
        session,
        # db_embedding_model_cls=BGEMultilingualGemma2ChunkEmbedding,
        db_embedding_model_cls=OpenAITextEmbedding3LargeChunkEmbedding,
    )

    documents_folder = f"{DATA_FOLDER}/accords_entreprise_niveau2"
    document_paths: list[Path] = []
    for prefix in INPUT_FILE_PREFIXES:
        document_paths.extend(
            Path(file) for file in glob.glob(f"{documents_folder}/{prefix}*")
        )

    doc_ingester = DocumentIngester(
        document_chunk_db_client=document_chunk_db_client,
        chunker=chunker,
        embedding_client=embedding_client,
    )
    doc_ingester.ingest_documents(document_paths=document_paths)


if __name__ == "__main__":
    main()
