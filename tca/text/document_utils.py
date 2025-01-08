from pathlib import Path

import docx
import textract
from PyPDF2 import PdfReader


class DocumentUtils:
    @staticmethod
    def extract_text_from_document(document_file: Path):
        if document_file.suffix == ".pdf":
            with open(document_file, "rb") as file:
                reader = PdfReader(file)
                return "\n".join(
                    [reader.pages[i].extract_text() for i in range(len(reader.pages))]
                )
        elif document_file.suffix == ".doc":
            return textract.process(str(document_file)).decode("utf-8")
        elif document_file.suffix == ".docx":
            doc = docx.Document(str(document_file))
            return "\n".join([para.text for para in doc.paragraphs])
        return ""
