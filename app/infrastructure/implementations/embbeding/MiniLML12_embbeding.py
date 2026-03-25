from sentence_transformers import SentenceTransformer
from app.core.interfaces.embbeding import EmbeddingStrategy
import json

class MiniLML12_Embbeding(EmbeddingStrategy):

    def __init__(self):
        self._model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device="cpu")

    @property
    def transformer(self):
        return self._model
    
    def embbed_it(self, chunks: list[str]):
        return self.transformer.encode(sentences=chunks)
    
    def embbed_it_for_model(self, chunks: list[str]):
        """
        Convert a list of text strings into embedding vectors for semantic search.
        
        Args:
            chunks: List of text strings to embed
            
        Returns:
            JSON string containing the embedding vectors
        """
        vectors = self.transformer.encode(sentences=chunks)
        return json.dumps(vectors.tolist())  # must return a string