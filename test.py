import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.io import read_image

import os
import argparse
from pathlib import Path
from shutil import copyfile, rmtree
from model import VGG13
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Define
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
CLASSES = ['cat', 'dog']
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Define model
model = VGG13().eval().to(DEVICE)
model.load_state_dict(torch.load('Models/E12_L0.0873_A0.97.pth'))

def main(args):
    # Testing on single Image
    if args.image:
        # Read image
        try:
            image = read_image(args.image)
        except:
            print(f"[ERROR]: Image corrupted!")
            exit()

        # Prediction
        inputs = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

        # Show result
        plt.imshow(image.permute(1, 2, 0))
        plt.title(CLASSES[pred], fontsize=30)
        plt.axis(False)
        plt.show()

    elif args.folder:
        # Output folder
        OUTPUT_FOLDER = Path('Results')

        # Check if output folder already exist
        if os.path.exists(OUTPUT_FOLDER):
            choice = input(f"'{OUTPUT_FOLDER}' folder existed! Replace (y/[n]): ")
            if choice.lower().strip() in ['y', 'yes']:
                rmtree(OUTPUT_FOLDER)
        
        # Create output folder
        for class_name in CLASSES:
            OUTPUT_FOLDER.joinpath(class_name).mkdir(parents=True, exist_ok=True)

       # Load Data folder
        DATASET = []
        for extention in ['jpg', 'jpeg', 'png']:
            DATASET.extend(list(Path(args.folder).rglob(f'*.{extention}')))

        # Loop through Data folder
        for image_path in tqdm(DATASET, 'Progress'):
            # Read image
            try:
                image = read_image(str(image_path))
            except:
                OUTPUT_FOLDER.joinpath('Corrupted').mkdir(parents=True, exist_ok=True)
                output_path = f"{OUTPUT_FOLDER}/Corrupted/{str(image_path).split('/')[-1]}"
                copyfile(image_path, output_path)
                continue

            # Prediction
            inputs = TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)

            output_path = f"{OUTPUT_FOLDER}/{CLASSES[pred]}/{str(image_path).split('/')[-1]}"
            if not os.path.exists(output_path):
                copyfile(image_path, output_path)

    else:
        # Load Dataset
        dataset = ImageFolder('PetImages', T.ToTensor())
        data_loader = DataLoader(dataset, 1, False, num_workers=os.cpu_count())

        # Predict amount = rows * cols
        rows, cols = 4, 4

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(11, 9), num=f"{rows*cols} prediction")

        for i in range(1, rows * cols + 1):
            # Select random image
            idx = torch.randint(0, len(data_loader), size=[1]).item()
            image, label = data_loader.dataset[idx]

            # Prediction
            inputs = TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)

            # Plot
            fig.add_subplot(rows, cols, i)
            plt.imshow(image.permute(1, 2, 0))
            if pred == label:
                plt.title(CLASSES[pred], fontsize=16, c='g')
            else:
                plt.title(CLASSES[pred], fontsize=16, c='r')
            plt.axis(False)
        plt.show()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing')
    args.add_argument('-i', '--image', type=str, default=None, help='Test on a single image (Default: None)')
    args.add_argument('-f', '--folder', type=str, default=None, help='Test on a folder of image (Default: None)')
    args.add_argument('-r', '--random', type=int, default=True, help='Random test in the Dataset')
    args = args.parse_args()
    
    main(args)
    