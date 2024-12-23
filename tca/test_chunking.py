import glob
import hashlib
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tca.document_chunk_manager import DocumentChunkManager

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost/{POSTGRES_DB}",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
)
Session = sessionmaker(bind=engine)
session = Session()

session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
session.execute(text("TRUNCATE TABLE document_chunks"))
session.execute(text("ALTER SEQUENCE document_chunks_id_seq RESTART WITH 1"))


document_chunk_manager = DocumentChunkManager(session)

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
documents_folder = f"{DATA_FOLDER}/accords_entreprise_test"
document_files = glob.glob(os.path.join(documents_folder, "*.txt"))

for document_file in document_files:
    with open(document_file, "r") as file:
        document_text = file.read()
        document_id = hashlib.sha256(document_text.encode()).hexdigest()
        doc_chunks = document_chunk_manager.chunk_document(document_text)
        for i, chunk_text in enumerate(doc_chunks):
            chunk_embedding = document_chunk_manager.get_embedding(chunk_text)
            document_chunk_manager.add_document_chunk(
                document_id=document_id,
                document_name=os.path.basename(document_file),
                chunk_text=chunk_text,
                chunk_embedding=chunk_embedding,
                extra_metadata={"chunk_index": i},
            )

results = document_chunk_manager.query_similar_chunks(
    query_embedding=[0.1, 0.2, 0.3], cos_dist_threshold=0.2
)

for result in results:
    print(result)
