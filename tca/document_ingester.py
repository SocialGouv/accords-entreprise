import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.document_chunk_manager import DocumentChunkManager
from tca.text.document_utils import DocumentLoader


class DocumentIngester:
    def __init__(self, document_chunk_manager: DocumentChunkManager):
        self.document_chunk_manager = document_chunk_manager

    def ingest_documents(self, document_paths: list[Path]) -> None:
        for document_path in document_paths:
            logging.info('Processing document "%s"', document_path)
            document_loaders = DocumentLoader()
            document_text = document_loaders.load_text_from_document(document_path)
            document_id = hashlib.sha256(document_text.encode()).hexdigest()
            logging.info('Extracting chunks from document "%s"', document_path)
            doc_chunks = self.document_chunk_manager.chunk_document(document_text)
            logging.info(
                'Generating embeddings for chunks of document "%s" and ingesting them in the Vector DB',
                document_path,
            )
            doc_chunks = [chunk for chunk in doc_chunks if chunk.strip()]
            all_chunk_embeddings = self.document_chunk_manager.generate_embedding(
                doc_chunks
            )
            for i, chunk_embeddings in enumerate(all_chunk_embeddings):
                self.document_chunk_manager.add_document_chunk(
                    document_id=document_id,
                    document_name=os.path.basename(document_path),
                    chunk_text=doc_chunks[i],
                    chunk_embeddings=chunk_embeddings,
                    extra_metadata={"chunk_index": i},
                )
