from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def __init__(self, conn_params: str|dict, collection: str):
        pass

    @abstractmethod
    def get_docs(self, query, params):
        pass