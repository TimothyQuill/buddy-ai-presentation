from search import CustomEmbeddingSearch

import pandas as pd


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/restaurant_ratings_enhanced.csv')

# E.g. 1, Find new foods a user will like given their history
for customer_id in ['CUST1036', 'CUST1029', 'CUST1022']:

    searcher = CustomEmbeddingSearch()
    history = df.loc[df['customer_id'] == customer_id]
    results = searcher.search(history)

    print('----------------')
    print(f'Given user {customer_id} had the following dishes:')
    print(history["dish_name"].tolist())
    print('I recommend these new dishes:')
    print(results)
