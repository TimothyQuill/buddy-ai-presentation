"""
This file takes dish information from a CSV and generates
synthetic semantic data, then stores it in a ChromaDB collection.
"""

import pandas as pd
import chat
from database import ChromaDBManager

# Relative location of the data
data_path = "data/"


class RecipeCollectionBuilder:
    """
    A class that loads dish data from a CSV, generates synthetic recipes
    and embeddings, and stores them in a ChromaDB collection.
    """

    def __init__(self, csv_name: str):
        """
        Initializes the RecipeCollectionBuilder with paths needed to load the data
        and create/store documents in ChromaDB.

        Args:
            csv_name (str): The base name of the CSV file (without extension).
        """
        self.csv_name = csv_name
        self.df = None
        self.manager = None
        self.persist_directory = ""

    def build_collection(self) -> None:
        """
        Main entry point to load the data from CSV, build the documents,
        and store them in the ChromaDB collection.
        """
        self.load_csv()
        self.make_documents()

    def create_document(self, dish: pd.Series, idx: int) -> dict:
        """
        Constructs a document containing a unique ID, the recipe text, its embedding,
        and metadata derived from a row in the DataFrame.

        Args:
            dish (pd.Series): A row from the DataFrame representing a single dish.
            idx (int): The index of the current dish, used to form a unique ID.

        Returns:
            dict: A dictionary with keys:
                  - "id": unique dish ID (str),
                  - "recipe": the generated recipe text (str),
                  - "embedding": the embedding vector (list of floats),
                  - "metadata": a dict of dish metadata (dict).
        """
        metadata = {
            "dish_name":    dish['dish_name'],
            "dish_desc":    dish['dish_description'],
            "cuisine_type": dish['cuisine_type'],
            "dietary_tags": dish['dietary_tags'],
        }

        # Generate the recipe text using dish metadata
        recipe_text = self.make_text(list(metadata.values()))

        # Create an embedding for the recipe text
        embedding = chat.generate_embedding(recipe_text)

        return {
            "id":        f"dish{idx}",
            "recipe":    recipe_text,
            "embedding": embedding,
            "metadata":  metadata
        }

    def load_csv(self) -> None:
        """
        Reads the CSV file into a pandas DataFrame and sets up a ChromaDB manager.
        """
        csv_file = f"{data_path}{self.csv_name}.csv"
        self.df = pd.read_csv(csv_file)

        # Create a ChromaDB manager with a collection name based on the CSV name
        self.manager = ChromaDBManager(
            collection_name=f"{self.csv_name}_collection",
            persist_directory=self.persist_directory
        )

    def make_documents(self) -> None:
        """
        Iterates over unique rows in the DataFrame (based on 'dish_name'),
        creates documents, and stores each one in the ChromaDB collection.
        """
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call load_csv() first.")

        # Identify unique rows based on 'dish_name' and get their indices
        unique_indexes = self.df.drop_duplicates(subset=["dish_name"], keep="first").index

        # For each unique dish, create and store its document in ChromaDB
        for i in unique_indexes:
            document = self.create_document(self.df.loc[i], i)
            self.store_document(document)

    @staticmethod
    def make_text(dish_attrs: list) -> str:
        """
            Generates a string using the provided arguments.

            Args:
                dish_attrs (list): A list of dish attributes
                                   (e.g., [dish_name, dish_desc, ...]).

                Returns:
                    str: The list of attributes.
                """
        return f"{dish_attrs}"

    def store_document(self, document: dict) -> None:
        """
        Stores a single document in the ChromaDB collection by adding its
        ID, recipe text, embedding, and metadata.

        Args:
            document (dict): A dict with keys "id", "recipe", "embedding", "metadata".
        """
        self.manager.add_embeddings(
            ids=[document["id"]],
            documents=[document["recipe"]],
            embeddings=[document["embedding"]],
            metadatas=[document["metadata"]]
        )

