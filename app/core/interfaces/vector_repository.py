from app.core.entities.documents import MilvusSchema
from typing import TypedDict
from abc import ABC, abstractmethod

class Entity(TypedDict):
    text: str

class SearchItem(TypedDict):
    id: str
    distance: float
    entity: Entity

class VectorRepository(ABC):

    @property
    @abstractmethod
    def client(self):
        pass

    @abstractmethod
    def insert(self, collection: str, data: list[MilvusSchema]):
        pass

    @abstractmethod
    def search(self, collection: str, vector: list[float]) -> list[list[SearchItem]]:
        pass