from openai import OpenAI
from neo4j import GraphDatabase
from .base_retriever import BaseRetriever
from langchain_core.documents import Document
from typing import List
from functools import reduce

class Neo4jRetriever(BaseRetriever):
    client = OpenAI()

    def __init__(self, conn_params: str | dict, collection: str):
        self.collection = collection
        self.auth = (conn_params.get("username", ""), conn_params.get("password", ""))
        self.uri = conn_params.get("uri", "")

    def get_docs(self, query: str, params={}) -> List[Document]:
        response = self.client.embeddings.create(
            input=query, model="text-embedding-ada-002"
        )

        query_vector = response.data[0].embedding

        query = """
        CALL db.index.vector.queryNodes($collection, $num_docs, $embedding)
        YIELD node AS chunk, score
        """
        with GraphDatabase.driver(self.uri, auth=self.auth) as driver:
            records, summary, keys = driver.execute_query(
                query,
                {
                    "collection": self.collection,
                    "num_docs": params.get("num_docs", 5),
                    "embedding": query_vector,
                },
                database_="neo4j",
            )

        metadata_keys = records[0]['chunk'].keys() -  ['text', 'embedding']

        documents = [
            Document(
                page_content=record["chunk"]["text"],
                metadata=reduce(
                    lambda a, x: {**a, **x},
                    [{key: record["chunk"][key]} for key in metadata_keys],
                    {},
                ),
            )
            for record in records
        ]

        return documents
