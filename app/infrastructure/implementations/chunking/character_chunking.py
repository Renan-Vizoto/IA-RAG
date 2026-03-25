from langchain_text_splitters import CharacterTextSplitter
from app.core.interfaces.chunking import ChunkingStrategy
from app.infrastructure.configs import settings

class CharacterChunking(ChunkingStrategy):

    def __init__(self):
        self._splitter = CharacterTextSplitter(
        separator="\n",                       # Primary split point (paragraphs)
        chunk_size=settings.chunk_size,       # Max characters per chunk
        chunk_overlap=settings.chunk_overlap, # Overlap to avoid context loss at boundaries
        length_function=len,                  # Use len() for character count
    )

    @property 
    def splitter(self):
        return self._splitter

    def chunk_it(self,text:str) -> list[str]:
        return self.splitter.split_text(text)
    
