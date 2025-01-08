from pathlib import Path

from doc2docx import convert
from langchain_community.document_loaders.word_document import Docx2txtLoader
from PyPDF2 import PdfReader

from tca.constants import DATA_FOLDER


class DocumentLoader:
    def __init__(self):
        converted_docx_folder = Path(f"{DATA_FOLDER}/converted_docx")
        converted_docx_folder.mkdir(parents=True, exist_ok=True)

    def load_text_from_pdf_file(self, document_file: Path):
        with open(document_file, "rb") as file:
            reader = PdfReader(file)
            return "\n".join(
                [reader.pages[i].extract_text() for i in range(len(reader.pages))]
            )

    def load_text_from_docx_file(self, document_file: Path):
        docx_loader = Docx2txtLoader(document_file)
        documents = docx_loader.load()

        return "\n".join([doc.page_content for doc in documents])

    def load_text_from_doc_file(self, document_file: Path):
        new_document_name = document_file.stem
        # TODO: Handle the saving and potential caching + overwritting of the new document in a more sustainable way
        new_document_path = Path(
            f"{DATA_FOLDER}/converted_docx/{new_document_name}.docx"
        )
        convert(str(document_file), new_document_path)

        return self.load_text_from_docx_file(new_document_path)

    def load_text_from_document(self, document_file: Path):
        extractors = {
            ".pdf": self.load_text_from_pdf_file,
            ".docx": self.load_text_from_docx_file,
            ".doc": self.load_text_from_doc_file,
        }

        extractor = extractors.get(document_file.suffix)
        if extractor:
            return extractor(document_file)
        return ""
