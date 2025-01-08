import glob
import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.constants import DATA_FOLDER
from tca.document_chunk_manager import DocumentChunkManager
from tca.text.document_utils import DocumentUtils

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


class DocumentIngester:
    def __init__(self, document_chunk_manager: DocumentChunkManager):
        self.document_chunk_manager = document_chunk_manager

    def ingest_documents(self) -> None:
        documents_folder = f"{DATA_FOLDER}/accords_entreprise_niveau2"
        document_paths: list[Path] = []
        for prefix in INPUT_FILE_PREFIXES:
            document_paths.extend(
                Path(file) for file in glob.glob(f"{documents_folder}/{prefix}*")
            )

        for document_path in document_paths:
            logging.info('Processing document "%s"', document_path)
            document_text = DocumentUtils.extract_text_from_document(document_path)
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
