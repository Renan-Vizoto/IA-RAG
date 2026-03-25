from abc import ABC, abstractmethod
from app.core.interfaces.extractor import IExtractor

class IFactory(ABC):

    @abstractmethod
    def create(self, type: str) -> IExtractor | None:
        pass
