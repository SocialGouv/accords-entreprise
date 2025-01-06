import glob
import hashlib
import logging
import logging.config
import os
from pathlib import Path

from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.embedding.embedding_clients import OllamaEmbeddingClient
from tca.text.document_utils import DocumentUtils

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
INPUT_FILE_PREFIXES = [
    "T09224067466",
    # "T00624061516",
    # "T07624061950",
    # "T04524061140",
]


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
