if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms

    import os.path

    ############################################################################################ Datasets

    dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
    valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                            train=True,
                                                            transform=None,  # transforms.ToTensor(),
                                                            download=True)

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                     transform=None,  # transforms.ToTensor(),
                                                     train=False)