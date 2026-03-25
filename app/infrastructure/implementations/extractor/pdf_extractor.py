from typing import BinaryIO
from pypdf import PdfReader
from app.core.interfaces.extractor import IExtractor

class PDFExtractor(IExtractor):

    def process(self, file: BinaryIO) -> list[str]:
        reader = PdfReader(file)
        text: list[str] = []
        for page in reader.pages:
            text.append(page.extract_text(0))

        return text