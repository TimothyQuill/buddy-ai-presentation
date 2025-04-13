"""
This file takes dish information from a CSV and generates
synthetic semantic data, then stores it in a ChromaDB collection.

Inherits from the base RecipeCollectionBuilder class and overrides
the make_text() method to perform an AI-generated recipe response.
"""

import os
import sys
import chat

# Get the directory path for the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Compute the parent directory path
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path so Python can find generate.py
sys.path.append(parent_dir)

from generate import RecipeCollectionBuilder  # <-- import the base class


# Note: 'generate.py' should contain the RecipeCollectionBuilder code
# from your second snippet, unchanged except for any needed paths.


class AIRecipeCollectionBuilder(RecipeCollectionBuilder):
    """
    Extends the base RecipeCollectionBuilder to use an AI approach
    for generating text instead of a simple string representation.
    """

    @staticmethod
    def make_text(dish_attrs: list) -> str:
        """
        Overridden method that uses the 'chat.generate_response' function
        to generate a recipe description based on the provided dish attributes
        and the template prompt.

        Args:
            dish_attrs (list): A list of dish attributes, which will be injected
                               into the prompt (e.g., [dish_name, dish_desc, ...]).

        Returns:
            str: A generated recipe text from the language model.
        """
        prompt = """
            Produce a recipe for a dish with the following info: 
                Name: {0} 
                Description: {1}
                Cuisine type: {2}
                Dietary tags: {3}
        """
        return chat.generate_response(prompt.format(*dish_attrs))


if __name__ == "__main__":
    # Build multiple collections from different CSV files
    # using the new AIRecipeCollectionBuilder
    for csv_name in ["cold_start_restaurant_ratings", "restaurant_ratings_enhanced"]:
        builder = AIRecipeCollectionBuilder(csv_name)

        # Change the location of the db to this directory
        builder.persist_directory = "using_recipe_embeddings"

        builder.build_collection()
