import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

import os
from model import VGG13
import matplotlib.pyplot as plt


# Define
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['cat', 'dog']
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load Dataset
dataset = ImageFolder('PetImages', T.ToTensor())
data_loader = DataLoader(dataset, 1, False, num_workers=os.cpu_count())

# Define model
model = VGG13().eval().to(DEVICE)
model.load_state_dict(torch.load('Models/E12_L0.0873_A0.97.pth'))

# Predict amount = rows * cols
rows, cols = 4, 4

plt.style.use('dark_background')
fig = plt.figure(figsize=(11, 9), num=f"{rows*cols} prediction")

for i in range(1, rows * cols + 1):
    # Select random image
    idx = torch.randint(0, len(data_loader), size=[1]).item()
    image, label = data_loader.dataset[idx]

    # Prediction
    input = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        outputs = model(input)
        _, pred = torch.max(outputs, 1)

    # Plot
    fig.add_subplot(rows, cols, i)
    plt.imshow(image.permute(1, 2, 0))
    if pred == label:
        plt.title(CLASSES[pred], fontsize=12, c='g')
    else:
        plt.title(CLASSES[pred], fontsize=12, c='r')
    plt.axis(False)
plt.show()
