from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

class EmbeddingStrategy(ABC):

    @property
    @abstractmethod
    def transformer(self) -> SentenceTransformer:
        pass

    @abstractmethod
    def embbed_it(self, chunks: list[str]):
        pass
    
    @abstractmethod
    def embbed_it_for_model(self, chunks: list[str]):
        pass