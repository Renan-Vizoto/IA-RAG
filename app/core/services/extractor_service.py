from app.core.interfaces.factory import IFactory
from io import BytesIO

class ExtractTextService:

    def __init__(self, extractor_factory: IFactory):
        self._factory = extractor_factory
    
    def process(self, file: BytesIO, file_type: str) -> list[str] | None:
        extractor = self._factory.create(file_type)
        if not extractor: 
            return None
        return extractor.process(file)
