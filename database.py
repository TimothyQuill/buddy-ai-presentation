import chromadb
from typing import List, Any, Dict

class ChromaDBManager:
    """
    A simple class to manage a Chroma database collection.
    Allows adding documents/embeddings, querying for similar embeddings,
    and filtering documents by metadata.
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
        # Create or open a persistent Chroma database at the given path:
        self.client = chromadb.PersistentClient(path=persist_directory)
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
            ids (List[str]): Unique IDs for each document.
            documents (List[str]): Textual content for each document.
            embeddings (List[List[float]]): Embedding vectors, one per document.
            metadatas (List[Dict[str, Any]], optional): Arbitrary metadata for each document.
                Defaults to an empty dictionary if None is provided.
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
            query_embedding (List[float]): A single embedding vector to search.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 5.

        Returns:
            Dict: A dictionary containing `ids`, `embeddings`, `documents`, and `metadatas`.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results

    def get_documents_by_metadata(self, key: str, value: Any) -> dict:
        """
        Retrieves documents whose metadata field `key` matches `value`,
        including the embedding vectors in the result.
        """
        results = self.collection.get(
            where={key: value},
            include=["embeddings", "documents", "metadatas"]
        )
        return results

    def persist(self) -> None:
        """
        For PersistentClient, data is written to disk upon each update.
        This method can be a no-op, or used for other finalizing logic.
        """
        # If your client or environment requires an explicit persist,
        # you can do that here. By default, PersistentClient
        # automatically saves changes.
        pass
