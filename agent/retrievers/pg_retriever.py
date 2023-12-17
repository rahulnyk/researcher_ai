from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings
from .base_retriever import BaseRetriever
from langchain_core.documents import Document
from typing import List

class PGRetriever(BaseRetriever):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    def __init__(self, conn_params: str|dict, collection: str):
        self.collection = collection
        self.conn_string = conn_params
        self.store = PGVector(
            collection_name=self.collection,
            connection_string=conn_params,
            embedding_function=self.embeddings,
        )

    def get_docs(self, query: str, options) -> List[Document]:
        result = self.store.max_marginal_relevance_search(query, **options)
        return result
