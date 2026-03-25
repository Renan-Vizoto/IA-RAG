from app.infrastructure.implementations.extractor.pdf_extractor import PDFExtractor
from app.core.interfaces.factory import IFactory
class ExtractorFactory(IFactory):
    _registry = {"pdf": PDFExtractor}

    def create(self, type: str):
        extractor = self._registry.get(type.lower())
        if not extractor:
            print(f"Unknown component: {type.lower()}")
            return None
        return extractor()
