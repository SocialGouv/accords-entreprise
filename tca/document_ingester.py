import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.database.document_chunk_db_client import DocumentChunkDBClient
from tca.embedding.embedding_clients import BaseEmbeddingClient
from tca.text.chunker import BaseChunker
from tca.text.document_utils import DocumentLoader


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
            logging.info('Processing document "%s"', document_path)
            document_loaders = DocumentLoader()
            document_text = document_loaders.load_text_from_document(document_path)
            document_id = hashlib.sha256(document_text.encode()).hexdigest()
            logging.info('Extracting chunks from document "%s"', document_path)
            doc_chunks = self.chunker.build_chunks(document_text)
            logging.info(
                'Generating embeddings for chunks of document "%s" and ingesting them in the Vector DB',
                document_path,
            )
            doc_chunks = [chunk for chunk in doc_chunks if chunk.strip()]
            all_chunk_embeddings = self.embedding_client.build_embedding(doc_chunks)
            for i, chunk_embeddings in enumerate(all_chunk_embeddings):
                self.document_chunk_db_client.add_document_chunk(
                    document_id=document_id,
                    document_name=os.path.basename(document_path),
                    chunk_text=doc_chunks[i],
                    chunk_embeddings=chunk_embeddings,
                    extra_metadata={"chunk_index": i},
                )
