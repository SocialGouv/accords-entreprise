import re
from pathlib import Path

from doc2docx import convert
from docx import Document
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
        doc = Document(str(document_file))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

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
            document_text = extractor(document_file)
            document_text = document_text.lower()
            # document_text = re.sub(r"\d+", "", document_text)
            # document_text = re.sub(r"[^a-zA-ZÀ-ÿ,.!?'\n\s\-]", "", document_text)
            # document_text = re.sub(r"(?<!\w)-(?!\w)", "", document_text)
            document_text = re.sub(r"\s+", " ", document_text)
            return document_text
        return ""
