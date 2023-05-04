import os
from pathlib import Path
from random import shuffle, seed
from shutil import copyfile, rmtree
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


def dataset_generator(config: dict):
    """Create Dataset folder"""

    # Set seed for shuffle data
    seed(config['random_seed'])

    # Get all classes and Dataset path
    classes = os.listdir(config["src_path"])
    data_path = Path(config["dest_path"])

    # Revmove old Dataset if replace=True
    rmtree(data_path) if config["replace"] and os.path.exists(data_path) else None

    data_dir = ['train']
    data_dir.append('val') if config["val_size"] != 0 else None
    data_dir.append('test') if config["test_size"] != 0 else None
    
    # Create Train, Val, Test folder for all classes
    for split_path in data_dir:
        for class_name in classes:
            data_path.joinpath(f"{split_path}/{class_name}").mkdir(parents=True, exist_ok=True)

    # Get all full path of data in source path
    dataset = list(Path(config["src_path"]).glob('*/*.jpg'))
    dataset_size = len(dataset)
    
    # Shuffle the dataset
    shuffle(dataset)
    
    # Calculate the train, val, test size
    val_split_size = int(config["val_size"] * dataset_size)
    test_split_size = int(config["test_size"] * dataset_size)
    train_split_size = dataset_size - (val_split_size + test_split_size)

    # Split the dataset
    train_data = dataset[:train_split_size]
    val_data = dataset[train_split_size:(train_split_size + val_split_size)]
    test_data = dataset[(train_split_size + val_split_size):]

    # Make function for copy file to new Dataset path
    def create_data(data, data_type: str):
        for src_path in tqdm(data, desc=f'Create {data_type} data'):
            dest_path = str(src_path).replace(config["src_path"], f"{data_path}/{data_type}")
            if not os.path.exists(dest_path):
                copyfile(src_path, dest_path)

    # Copy splitted data to dest_path
    create_data(train_data, 'train')
    create_data(val_data, 'val') if val_data else None
    create_data(test_data, 'test') if test_data else None


def create_dataloaders(data_path: str, batch_size: int=32, train_transform: T.Compose=None, num_worker: int=os.cpu_count()):
    # Transforms for Val and Test data from Train transforms
    if train_transform is not None:
        val_transforms = test_transforms = T.Compose([
            train_transform.transforms[0],  # size must be the first transform
            T.ToTensor()])
    else:
        val_transforms = test_transforms = None
    
    # Load Dataset folder
    train_data = ImageFolder(root=f"{data_path}/train", transform=train_transform)
    val_data = ImageFolder(root=f"{data_path}/val", transform=val_transforms) if os.path.exists(f"{data_path}/val") else None
    test_data = ImageFolder(root=f"{data_path}/test", transform=test_transforms) if os.path.exists(f"{data_path}/test") else None

    # Create Dataloader
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  num_workers=num_worker,
                                  shuffle=True)
    val_dataloader = DataLoader(val_data,
                                  batch_size=batch_size,
                                  num_workers=num_worker,
                                  shuffle=False) if val_data is not None else None
    test_dataloader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  num_workers=num_worker,
                                  shuffle=False) if test_data is not None else None


    return train_dataloader, val_dataloader, test_dataloader
