"""
This file creates a custom embedding for a given customer by aggregating dish embeddings.
It then uses the custom embedding to query another collection and returns matching dish names.
"""

import numpy as np
import pandas as pd
from database import ChromaDBManager


class CustomEmbeddingSearch:
    """
    A class that manages two ChromaDB collections (an 'enhanced' collection for retrieving
    dish embeddings, and a 'cold_start' collection for searching based on a custom user embedding).
    """

    def __init__(self, persist_directory: str = "using_recipe_embeddings/"):
        """
        Initializes ChromaDB managers for both collections using the specified directory.

        Args:
            persist_directory (str, optional): The directory path where the ChromaDB files
                                              are stored or will be created.
                                              Defaults to "using_recipe_embeddings/".
        """
        self.manager_enhanced = ChromaDBManager(
            collection_name="restaurant_ratings_enhanced_collection",
            persist_directory=persist_directory
        )
        self.manager_cold_start = ChromaDBManager(
            collection_name="cold_start_restaurant_ratings_collection",
            persist_directory=persist_directory
        )

    def get_embedding(self, row: pd.Series) -> list:
        """
        Retrieves the embedding for a specific dish from the enhanced collection.

        Args:
            row (pd.Series): A row from the user history DataFrame containing at least
                             a 'dish_name' field.

        Returns:
            list: The embedding vector for the dish.
        """
        # Get the document using metadata filtering on 'dish_name'
        document = self.manager_enhanced.get_documents_by_metadata("dish_name", row['dish_name'])
        # Extract the first embedding from the returned result
        embedding = document['embeddings'][0]
        return embedding

    def create_custom_embedding(self, history: pd.DataFrame) -> list:
        """
        Creates a custom user embedding by summing the embeddings for each dish in the user's history.

        Args:
            history (pd.DataFrame): A DataFrame containing rows with dish data (including 'dish_name').

        Returns:
            list: The custom aggregated embedding (as a Python list).
        """
        # Initialize a zero vector of the size expected by the embedding model
        # (e.g., text-embedding-ada-002 has 1536 dimensions)
        user_embedding = np.zeros(1536, dtype=np.float32)

        # Loop over each row in the user's history DataFrame
        for _, row_data in history.iterrows():
            # Retrieve the dish embedding for the current row
            dish_embedding = self.get_embedding(row_data)
            # Sum the dish embedding into the aggregate user embedding
            user_embedding += dish_embedding

        # Convert the numpy array to a list before returning
        return user_embedding.tolist()

    def search(self, history: pd.DataFrame, k: int = 5) -> list:
        """
        Searches the cold start collection based on a custom embedding created from the user history.

        Args:
            history (pd.DataFrame): A DataFrame containing the user's dish history.
            k (int, optional): The number of nearest documents to return from the cold start collection.
                               Defaults to 100.

        Returns:
            list: A list of dish names that match the custom embedding criteria.
        """
        # Create a custom embedding based on the user's history
        custom_embedding = self.create_custom_embedding(history)
        # Use the custom embedding to query the cold start collection
        results = self.manager_cold_start.query_embeddings(custom_embedding, k=k)
        # Extract the metadata from the query results (assumes a structure with a list of metadata dicts)
        metadatas = results['metadatas'][0]
        # Extract the 'dish_name' from each metadata dictionary to obtain the matching dish names
        dish_names = [entry["dish_name"] for entry in metadatas]

        return dish_names
