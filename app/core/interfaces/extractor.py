from abc import ABC, abstractmethod
from typing import BinaryIO

class IExtractor(ABC):
    
    @abstractmethod
    def process(self, file: BinaryIO) -> list[str]:
        pass

