import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.database.document_chunk_db_client import DocumentChunkDBClient
from tca.embedding.embedding_clients import BaseEmbeddingClient
from tca.text.chunker import BaseChunker
from tca.text.document_utils import DocumentLoader

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class DocumentIngester:
    def __init__(
        self,
        document_chunk_db_client: DocumentChunkDBClient,
        chunker: BaseChunker,
        embedding_client: BaseEmbeddingClient,
    ):
        self.document_chunk_db_client = document_chunk_db_client
        self.chunker = chunker
        self.embedding_client = embedding_client

    def ingest_documents(self, document_paths: list[Path]) -> None:
        for document_path in document_paths:
            logger.info('Processing document "%s"', document_path)
            document_loaders = DocumentLoader()
            document_text = document_loaders.load_text_from_document(document_path)
            document_id = hashlib.sha256(document_text.encode()).hexdigest()
            logger.info('Extracting chunks from document "%s"', document_path)
            doc_chunks = self.chunker.build_chunks(document_text)
            logger.info(f"Number of chunks built: {len(doc_chunks)}")
            logger.info(
                'Generating embeddings for chunks of document "%s" and ingesting them in the Vector DB',
                document_path,
            )
            all_chunk_embeddings = self.embedding_client.encode_corpus(doc_chunks)
            for i, chunk_embeddings in enumerate(all_chunk_embeddings):
                self.document_chunk_db_client.add_document_chunk(
                    document_id=document_id,
                    document_name=os.path.basename(document_path),
                    chunk_text=doc_chunks[i],
                    chunk_embeddings=chunk_embeddings,
                    extra_metadata={"chunk_index": i},
                )
