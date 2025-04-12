import chromadb
from chromadb.config import Settings
from typing import List, Any, Dict


class ChromaDBManager:
    """
    A simple class to manage a Chroma database collection.
    Allows adding documents/embeddings and querying for similar embeddings.
    """

    def __init__(self, collection_name: str = "my_collection", persist_directory: str = ".chroma_db"):
        """
        Initializes the ChromaDBManager with a given collection name and optional persistence directory.

        Args:
            collection_name (str, optional): The name of the collection to create or retrieve.
                Defaults to "my_collection".
            persist_directory (str, optional): The directory where Chroma data will be stored.
                Defaults to ".chroma_db".
        """
        # Create a Chroma client with optional persistence
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            )
        )
        # Create or retrieve the specified collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embeddings(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None
    ) -> None:
        """
        Adds documents, embeddings, and optional metadata to the Chroma collection.

        Args:
            ids (List[str]): Unique IDs for each embedding/document.
            documents (List[str]): The textual data corresponding to each embedding.
            embeddings (List[List[float]]): A list of embeddings, one per document.
            metadatas (List[Dict[str, Any]], optional): Metadata for each document
                (e.g., source, tags). Defaults to None.
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_embeddings(self, query_embedding: List[float], k: int = 5) -> Dict:
        """
        Finds the k closest embeddings in the collection to the given query embedding.

        Args:
            query_embedding (List[float]): The embedding vector to search for.
            k (int, optional): The number of closest matches to retrieve. Defaults to 5.

        Returns:
            Dict: A dictionary containing `ids`, `embeddings`, `documents`, and `metadatas`.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results

    def persist(self) -> None:
        """
        Persists the data in the Chroma client to the specified directory (if any).
        Call this if you want to ensure data is saved between sessions.
        """
        self.client.persist()
