import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import json


def format_dataset(row):
    finetuning_block = {
        "messages": [
            {
                "role": "user",
                "content": f"""
        You are provided with an image that has been converted to ASCII art and your task is to classify it among 10 categories.
        Here are the possible categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
        
        You MUST only output the identified class and nothing else.
        
        ### ASCII art image:
        {row['text']}
        
        ### Response:
        """,
            },
            {"role": "assistant", "content": str(row["target"])},
        ]
    }
    return finetuning_block

def worker(dataset, start, end, result_queue):
    dataset_formatted = []
    for index, row in dataset[start:end].iterrows():
        dataset_formatted.append(format_dataset(row))
    result_queue.put(dataset_formatted)

def generate_data(train_dataset: pd.DataFrame, valid_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
    num_processes = mp.cpu_count()  # Get number of processes

    # Create queues to store the results
    train_queue = mp.Queue()
    valid_queue = mp.Queue()
    test_queue = mp.Queue()

    # Create and start the processes
    processes = []

    # Train dataset processes
    chunk_size = (train_dataset.shape[0] + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(train_dataset, start, end, train_queue))
        processes.append(process)
        process.start()

    # Valid dataset processes
    chunk_size = (valid_dataset.shape[0] + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(valid_dataset, start, end, valid_queue))
        processes.append(process)
        process.start()

    # Test dataset processes
    chunk_size = (test_dataset.shape[0] + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(test_dataset, start, end, test_queue))
        processes.append(process)
        process.start()

    # Collect the results
    train_dataset_formatted, valid_dataset_formatted, test_dataset_formatted = [] , [], []

    for _ in tqdm(range(num_processes)):
        train_dataset_prompts = train_queue.get()
        valid_dataset_prompts = valid_queue.get()
        test_dataset_prompts = test_queue.get()

        train_dataset_formatted.extend(train_dataset_prompts)
        valid_dataset_formatted.extend(valid_dataset_prompts)
        test_dataset_formatted.extend(test_dataset_prompts)

    for process in processes:
        process.join()

    return train_dataset_formatted, valid_dataset_formatted, test_dataset_formatted


def save_as_jsonl(train_dataset_formatted, valid_dataset_formatted, test_dataset_formatted):

    train_dataset_final, test_dataset_final = train_dataset_formatted + valid_dataset_formatted, test_dataset_formatted[:150]

    with open("transformed_data/train.jsonl", "w") as f:
        for line in train_dataset_final:
            json.dump(line, f)
            f.write("\n")

    with open("transformed_data/test.jsonl", "w") as f:
        for line in test_dataset_final:
            json.dump(line, f)
            f.write("\n")


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = (pd.read_csv('transformed_data/train.csv', index_col=0),
                                                  pd.read_csv('transformed_data/valid.csv', index_col=0),
                                                  pd.read_csv('transformed_data/test.csv', index_col=0))

    train_dataset, valid_dataset, test_dataset = generate_data(train_dataset, valid_dataset, test_dataset)

    save_as_jsonl(train_dataset, valid_dataset, test_dataset)

    print("------ Done and saved --------")

