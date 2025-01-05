import glob
import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager
from tca.text.document_utils import DocumentUtils

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

postgres_session_manager = PostgresSessionManager()
session = postgres_session_manager.session


document_chunk_manager = DocumentChunkManager(session)

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
INPUT_FILE_PREFIXES = [
    "T09224067466",
    "T00624061516",
    "T07624061950",
    "T04524061140",
]


# TODO: Then implement embedding with a local embedding model
# TODO: This early test should show that the pipeline works. Store the result so it can be compared to our labeled data
# TODO: Finally, implement the real embedding and chunking logic using a good embedder and chunking with langchain or similar
def main() -> None:
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
        doc_chunks = document_chunk_manager.chunk_document(document_text)
        logging.info(
            'Generating embeddings for chunks of document "%s" and ingesting them in the Vector DB',
            document_path,
        )
        for i, chunk_text in enumerate(doc_chunks):
            if not chunk_text.strip():
                continue
            chunk_embedding = document_chunk_manager.generate_embedding(chunk_text)
            document_chunk_manager.add_document_chunk(
                document_id=document_id,
                document_name=os.path.basename(document_path),
                chunk_text=chunk_text,
                chunk_embedding=chunk_embedding,
                extra_metadata={"chunk_index": i},
            )

    # results = document_chunk_manager.query_similar_chunks(
    #     query_embedding=[0.1, 0.2, 0.3], cos_dist_threshold=0.2
    # )

    # for result in results:
    #     print(result)


if __name__ == "__main__":
    main()
