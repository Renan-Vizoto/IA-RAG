from abc import ABC, abstractmethod

class SchemaBuilder(ABC):

    @property
    @abstractmethod
    def client(self):
        pass
    
    @abstractmethod
    def build(self, name: str):
        pass