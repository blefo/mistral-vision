import torch
import torchvision
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from tqdm import tqdm
import multiprocessing as mp


# Function to convert an image to ASCII
def image_to_ascii(image: Image.Image, width: int = 100, height: int = 50) -> str:
    gray_scale = "&()!$€£/;@%#*+=-:. "
    image = image.resize((width, height))
    image = image.convert("L")  # Convert to grayscale

    pixels = np.array(image)
    ascii_str = ""

    for pixel_row in pixels:
        for pixel in pixel_row:
            ascii_str += gray_scale[pixel // 32]
        ascii_str += "\n"

    return ascii_str


# Custom transform to convert image to ASCII art
class ToAscii:
    def __call__(self, image):
        ascii_art = image_to_ascii(image)
        return ascii_art


def load_data(dataset_name: str, transform: bool = False):
    dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', dataset_name)

    valid_ratio = 0.2  # Using 80%/20% split for train/valid

    if transform:
        # First set of transformations that work with PIL images
        pil_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.1)
        ])

        # Second set of transformations that work with tensors
        tensor_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        pil_transforms = v2.ToImage()
        tensor_transforms = v2.ToTensor()  # Convert image to tensor if no transformations are specified

    if dataset_name == "FashionMINST":
        # Load the dataset for the training/validation sets
        train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                                train=True,
                                                                transform=None,
                                                                download=True)

        # Split it into training and validation sets
        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
        nb_valid = int(valid_ratio * len(train_valid_dataset))
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

        # Load the test set
        test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                         transform=None,
                                                         train=False)
    elif dataset_name == "MNIST":
        # Load the dataset for the training/validation sets
        train_valid_dataset = torchvision.datasets.MNIST(root=dataset_dir,
                                                                train=True,
                                                                transform=None,
                                                                download=True)

        # Split it into training and validation sets
        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
        nb_valid = int(valid_ratio * len(train_valid_dataset))
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

        # Load the test set
        test_dataset = torchvision.datasets.MNIST(root=dataset_dir,
                                                         transform=None,
                                                         train=False)


    # Apply the transformations on the fly in the dataset access
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, pil_transform, tensor_transform, ascii_transform):
            self.dataset = dataset
            self.pil_transform = pil_transform
            self.tensor_transform = tensor_transform
            self.ascii_transform = ascii_transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx, minimalist: bool = False):
            image, label = self.dataset[idx]
            image = self.pil_transform(image)
            ascii_art = self.ascii_transform(image)
            if not minimalist:
                transformed_image = self.tensor_transform(F.to_tensor(image))
                return (ascii_art, transformed_image), label
            else:
                return ascii_art, label

    train_dataset = CustomDataset(train_dataset, pil_transforms, tensor_transforms, ToAscii())
    valid_dataset = CustomDataset(valid_dataset, pil_transforms, tensor_transforms, ToAscii())
    test_dataset = CustomDataset(test_dataset, pil_transforms, tensor_transforms, ToAscii())

    return train_dataset, valid_dataset, test_dataset

def worker(dataset, start, end, result_queue):
    dataset_text, dataset_target = [], []
    for i in range(start, min(end, dataset.__len__())):
        item_val = dataset.__getitem__(i, minimalist=True)
        dataset_text.append(item_val[0])
        dataset_target.append(item_val[1])
    result_queue.put((dataset_text, dataset_target))

def convert_and_save_to_dataframe_multiprocessing(train_dataset, valid_dataset, test_dataset):
    train_dataset_len = train_dataset.__len__()
    valid_dataset_len = valid_dataset.__len__()
    test_dataset_len = test_dataset.__len__()

    num_processes = mp.cpu_count()  # Get the number of available cores

    # Create queues to store the results
    train_queue = mp.Queue()
    valid_queue = mp.Queue()
    test_queue = mp.Queue()

    # Create and start the processes
    processes = []

    # Train dataset processes
    chunk_size = (train_dataset_len + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(train_dataset, start, end, train_queue))
        processes.append(process)
        process.start()

    # Valid dataset processes
    chunk_size = (valid_dataset_len + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(valid_dataset, start, end, valid_queue))
        processes.append(process)
        process.start()

    # Test dataset processes
    chunk_size = (test_dataset_len + num_processes - 1) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        process = mp.Process(target=worker, args=(test_dataset, start, end, test_queue))
        processes.append(process)
        process.start()

    # Collect the results from the queues
    train_dataset_text, train_dataset_target = [], []
    valid_dataset_text, valid_dataset_target = [], []
    test_dataset_text, test_dataset_target = [], []

    for _ in tqdm(range(num_processes)):
        train_text, train_target = train_queue.get()
        valid_text, valid_target = valid_queue.get()
        test_text, test_target = test_queue.get()

        train_dataset_text.extend(train_text)
        train_dataset_target.extend(train_target)
        valid_dataset_text.extend(valid_text)
        valid_dataset_target.extend(valid_target)
        test_dataset_text.extend(test_text)
        test_dataset_target.extend(test_target)

    for process in processes:
        process.join()

    train_dataset_dict = {'text': train_dataset_text, 'target': train_dataset_target}
    valid_dataset_dict = {'text': valid_dataset_text, 'target': valid_dataset_target}
    test_dataset_dict = {'text': test_dataset_text, 'target': test_dataset_target}

    train_dataset_df = pd.DataFrame(train_dataset_dict)
    valid_dataset_df = pd.DataFrame(valid_dataset_dict)
    test_dataset_df = pd.DataFrame(test_dataset_dict)

    return train_dataset_df, valid_dataset_df, test_dataset_df

def convert_and_save_to_dataframe(train_dataset, valid_dataset, test_dataset):
    train_dataset_text, train_dataset_target = [], []
    valid_dataset_text, valid_dataset_target = [], []
    test_dataset_text, test_dataset_target = [], []
    train_dataset_df_len, valid_dataset_len, test_dataset_len = train_dataset.__len__(), valid_dataset.__len__(), test_dataset.__len__()
    max_size: int = max(train_dataset_df_len, valid_dataset_len, test_dataset_len)

    for i in tqdm(range(max_size)):
        if i <= train_dataset_df_len:
            item_val = train_dataset.__getitem__(i, minimalist=True)
            train_dataset_text.append(item_val[0])
            train_dataset_target.append(item_val[1])

        if i <= valid_dataset_len:
            item_val = valid_dataset.__getitem__(i, minimalist=True)
            valid_dataset_text.append(item_val[0])
            valid_dataset_target.append(item_val[1])

        if i <= test_dataset_len:
            item_val = test_dataset.__getitem__(i, minimalist=True)
            test_dataset_text.append(item_val[0])
            test_dataset_target.append(item_val[1])

    train_dataset_dict, valid_dataset_dict, test_dataset_dict = ({'text': train_dataset_text, 'target': train_dataset_target},
                                                                 {'text': valid_dataset_text, 'target': valid_dataset_target},
                                                                 {'text': test_dataset_text, 'target': test_dataset_target})

    train_dataset_df, valid_dataset_df, test_dataset_df = pd.DataFrame(train_dataset_dict),\
                                                           pd.DataFrame(valid_dataset_dict),\
                                                           pd.DataFrame(test_dataset_dict)

    return train_dataset_df, valid_dataset_df, test_dataset_df


if __name__ == '__main__':
    '''
        convert_and_save_to_dataframe_multiprocessing: 28 seconds with 39 cores
        convert_and_save_to_dataframe: 14 minutes focusing on one core
    '''
    train_dataset, valid_dataset, test_dataset = load_data("MNIST", True)

    train_dataset, valid_dataset, test_dataset = convert_and_save_to_dataframe_multiprocessing(train_dataset, valid_dataset, test_dataset)

    train_dataset.to_csv('transformed_data/train.csv')
    valid_dataset.to_csv('transformed_data/valid.csv')
    test_dataset.to_csv('transformed_data/test.csv')

    # Example of accessing an image and its ASCII transformation
    # for i in range(3):
    #     (ascii_art, transformed_image), label = train_dataset[i]
    #     print(f"Label: {label}")
    #     print("ASCII Art:")
    #     print(ascii_art)  # The image is now an ASCII art string
    #     print("Transformed Image Tensor:")
    #     print(transformed_image)  # The image tensor after transformations
    #     print("\n" + "=" * 50 + "\n")
