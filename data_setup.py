import os
from pathlib import Path
from shutil import copyfile, rmtree
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


def dataset_generator(
        src_path: str, 
        dest_path: str = 'Dataset', 
        split_size: list = [0.8, 0.2], 
        image_format: list = ['jpg']
):
    """Create Dataset folder"""

    if not os.path.exists(src_path):
        raise Exception('Source path not found!')

    # Get all classes and Dataset path
    CLASSES = os.listdir(src_path)
    DATA_PATH = Path(dest_path)

    # Revmove old Dataset if replace=True
    rmtree(DATA_PATH) if os.path.exists(DATA_PATH) else None
    
    # Create Train, Test folder for all classes
    for split_path in ['train', 'test']:
        for class_name in CLASSES:
            DATA_PATH.joinpath(f"{split_path}/{class_name}").mkdir(parents=True, exist_ok=True)

    # Get all full path of data in source path
    DATASET = []
    for format in image_format:
        DATASET.extend(list(Path(src_path).rglob(f'*/*.{format}')))

    # Split the dataset
    train_data, test_data = random_split(DATASET, split_size)

    # Make function for copy file to new Dataset path
    def create_data(data, data_type: str):
        for copy_path in tqdm(data, desc=f'Create {data_type} data'):
            paste_path = str(copy_path).replace(src_path, f"{DATA_PATH}/{data_type}")
            if not os.path.exists(paste_path):
                copyfile(copy_path, paste_path)

    # Copy splitted data to dest_path
    create_data(train_data, 'train')
    create_data(test_data, 'test')


def create_dataloaders(
        data_path: str = 'Dataset', 
        batch_size: int = 32, 
        train_transform: T.Compose = None, 
        test_transform: T.Compose = None, 
        num_worker: int = os.cpu_count()
):
    """Create and return train_loader, test_loader"""

    if not os.path.exists(data_path):
        raise Exception('Data path not found!')
    
    # Load Dataset folder
    train_data = ImageFolder(root=f"{data_path}/train", transform=train_transform)
    test_data = ImageFolder(root=f"{data_path}/test", transform=test_transform)

    # Create Dataloader
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=num_worker,
                              shuffle=True)
    
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             num_workers=num_worker,
                             shuffle=False)

    return train_loader, test_loader
