import mistralai
from mistralai.exceptions import MistralAPIStatusException, MistralException
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient

from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
import requests


def prompt(row):
    return f"""
        You are provided with an image that has been converted to ASCII art and your task is to classify it among 10 categories.
        Here are the possible categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

        You MUST only output the identified class and nothing else.

        ### ASCII art image:
        {row['text']}

        ### Response:
    """


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException,
                                MistralAPIStatusException,
                                MistralException))
def get_inference(index, llm_input_value: str):
    chat_response = client.chat(
        model=client.jobs.retrieve("be2e7a1c-21e3-458a-881c-9d9adca22cef").fine_tuned_model,
        messages=[ChatMessage(role='user', content=llm_input_value)]
    )
    return index, chat_response.choices[0].message.content


if __name__ == "__main__":
    api_key = "puueDsQhS9LXPdEnooegEai7NeT4oer4"  # This key has been revoked
    client = MistralClient(api_key=api_key)

    # Init the validation data
    validation_data = pd.read_csv('transformed_data/test.csv', index_col=0)[150:]
    validation_data = validation_data.reset_index(drop=True)
    validation_data = validation_data[:3000]
    validation_data['llm_input'] = validation_data[['text']].apply(lambda x: prompt(x), axis=1)

    results = []

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(get_inference, i, validation_data['llm_input'].loc[i]) for i in
                   range(validation_data.shape[0])]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # Sort results by index
    results.sort(key=lambda x: x[0])

    # Extract only the response part
    llm_results = pd.Series([result[1] for result in results], name="llm_response")

    # Concatenate with the original validation data
    validation_data = pd.concat([validation_data, llm_results], axis=1)

    print('''------ Saving File --------''')

    validation_data.to_csv('transformed_data/results_on_validation.csv')

    print('''------ Experiment Done --------''')
