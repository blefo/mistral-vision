import torch
import torchvision
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from tqdm import tqdm


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

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            image = self.pil_transform(image)
            ascii_art = self.ascii_transform(image)
            transformed_image = self.tensor_transform(F.to_tensor(image))
            return (ascii_art, transformed_image), label

    train_dataset = CustomDataset(train_dataset, pil_transforms, tensor_transforms, ToAscii())
    valid_dataset = CustomDataset(valid_dataset, pil_transforms, tensor_transforms, ToAscii())
    test_dataset = CustomDataset(test_dataset, pil_transforms, tensor_transforms, ToAscii())

    return train_dataset, valid_dataset, test_dataset


def convert_and_save_to_dataframe(train_dataset, valid_dataset, test_dataset):
    train_dataset_df_dict, valid_dataset_dict, test_dataset_dict = {'text': [], 'target': []}, {'text': [], 'target': []}, {'text': [], 'target': []}
    max_size: int = max(train_dataset.__len__(), valid_dataset.__len__(), test_dataset.__len__())

    for i in tqdm(range(max_size)):
        if train_dataset.__getitem__(i):
            train_dataset_df_dict['text'] = train_dataset.__getitem__(i)[0][0]
            train_dataset_df_dict['target'] = train_dataset.__getitem__(i)[1]

        if valid_dataset.__getitem__(i):
            valid_dataset_dict['text'] = valid_dataset.__getitem__(i)[0][0]
            valid_dataset_dict['target'] = valid_dataset.__getitem__(i)[1]

        if test_dataset.__getitem__(i):
            test_dataset_dict['text'] = test_dataset.__getitem__(i)[0][0]
            test_dataset_dict['target'] = test_dataset.__getitem__(i)[1]

    train_dataset_df, valid_dataset_df, test_dataset_df = pd.DataFrame(train_dataset_df_dict),\
                                                           pd.DataFrame(valid_dataset_dict),\
                                                           pd.DataFrame(test_dataset_dict)

    return train_dataset_df, valid_dataset_df, test_dataset_df


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = load_data("MNIST", True)

    train_dataset, valid_dataset, test_dataset = convert_and_save_to_dataframe(train_dataset, valid_dataset, test_dataset)

    # Example of accessing an image and its ASCII transformation
    for i in range(3):
        (ascii_art, transformed_image), label = train_dataset[i]
        print(f"Label: {label}")
        print("ASCII Art:")
        print(ascii_art)  # The image is now an ASCII art string
        print("Transformed Image Tensor:")
        print(transformed_image)  # The image tensor after transformations
        print("\n" + "=" * 50 + "\n")
