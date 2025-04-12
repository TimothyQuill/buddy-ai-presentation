from os import environ

import openai


# Replace this with your actual API key
openai.api_key = environ['OPENAI_KEY']


def generate_response(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
        Generates a conversational response using OpenAI's ChatCompletion API.

        This function sends the given prompt to the specified OpenAI chat model and
        returns the AI assistant's reply. It includes a system message to set a
        specific context or persona (here, a helpful assistant).

        Args:
            prompt (str): The text or query you want the AI assistant to respond to.
            model (str, optional): The GPT model to use for generating responses.
                Defaults to "gpt-3.5-turbo".

        Returns:
            str: The AI-generated response as a string. If any error occurs, the
            function returns an error message string containing the exception details.
        """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                # A system message is like setting the overall context or persona
                {"role": "system", "content": "You are a helpful assistant."},
                # The user message is the prompt or question
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4096
        )

        # Extract and return the assistant's reply
        assistant_message = response["choices"][0]["message"]["content"]
        return assistant_message

    except Exception as e:
        return f"An error occurred: {e}"


def generate_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generate an embedding for the input text using OpenAI's text-embedding-ada-002 model.

    Args:
        text (str): The input text you want to embed.
        model (str): Name of the embedding model. Defaults to text-embedding-ada-002.

    Returns:
        list: The embedding vector as a list of floats.
    """

    response = openai.Embedding.create(
        model=model,
        input=text
    )

    embedding = response['data'][0]['embedding']
    return embedding
