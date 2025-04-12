import chat
import pandas as pd
from recipe_prompt_template import prompt
import utils


data_path = '/Users/tim/PycharmProjects/buddy-ai-challenge/data/'
df = pd.read_csv(data_path+'cold_start_restaurant_ratings.csv')


def make_recipe(dish):
    return chat.generate_response(
        prompt.format(
            dish['dish_name'],
            dish['dish_description'],
            dish['cuisine_type'],
            dish['dietary_tags']
        )
    )


def generate_data():

    unique_indexes = df.drop_duplicates(subset=["dish_name"], keep="first").index

    for i in unique_indexes:
        recipe = make_recipe(df.loc[i])
        embedding = chat.generate_embedding(recipe)
        print(recipe)
        print(embedding, len(embedding))
