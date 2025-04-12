from database import ChromaDBManager


if __name__ == "__main__":
    # 1. Initialize the manager (this creates or retrieves the collection)
    manager = ChromaDBManager(collection_name="recipes_collection")

    # Example data
    recipe_texts = [
        "Chocolate cake with dark chocolate frosting.",
        "Vanilla ice cream with fresh strawberries.",
        "Spaghetti carbonara with pancetta and parmesan cheese."
    ]
    # Example embeddings (make sure you generate real embeddings in practice)
    recipe_embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.4, 0.2, 0.1],
        [0.9, 0.8, 0.2, 0.05]
    ]
    recipe_ids = ["recipe1", "recipe2", "recipe3"]
    recipe_metadatas = [
        {"dish": "chocolate_cake"},
        {"dish": "vanilla_ice_cream"},
        {"dish": "spaghetti_carbonara"}
    ]

    # 2. Add documents/embeddings to the collection
    manager.add_embeddings(
        ids=recipe_ids,
        documents=recipe_texts,
        embeddings=recipe_embeddings,
        metadatas=recipe_metadatas
    )

    # 3. Query the collection with a new embedding
    query_vector = [0.12, 0.18, 0.29, 0.38]  # hypothetical embedding
    results = manager.query_embeddings(query_embedding=query_vector, k=2)

    print("Query Results:", results)

    # 4. If you want to persist changes to disk, call:
    # manager.persist()