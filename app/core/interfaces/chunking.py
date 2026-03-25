from abc import ABC, abstractmethod
from langchain_text_splitters import TextSplitter

class ChunkingStrategy(ABC):

    @property
    @abstractmethod
    def splitter() -> TextSplitter:
        pass

    def chunk_it(self,text:str) -> list[str]:
        splitter = self._getSplitter()
        return splitter.split_text(text)